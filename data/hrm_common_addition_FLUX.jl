

module HRMFluxAddCarry

using Flux
using NNlib: gelu
using Zygote
using Random, Statistics
using PositionalEmbeddings  # AbsolutePE


"""
A minimal Transformer block compatible with (d, L, B) tensors.

Fields:
- positional_encoding_kind :: Symbol  (:none | :sinusoidal | :learned)
- positional_layer         :: Union{Nothing, PositionalEmbeddings.AbsolutePE, Flux.Embedding}
- attention                :: Flux.MultiHeadAttention
- norm_before_attention    :: Flux.LayerNorm
- feedforward              :: Flux.Chain
- norm_before_feedforward  :: Flux.LayerNorm
"""
struct TransformerBlock{PL}
    positional_encoding_kind::Symbol
    positional_layer::PL
    attention::Flux.MultiHeadAttention
    norm_before_attention::Flux.LayerNorm
    feedforward::Flux.Chain
    norm_before_feedforward::Flux.LayerNorm
end

"""
    TransformerBlock(d; nheads, ff_mult=4, attention_dropout_probability=0.0,
                        positional_encoding_kind=:sinusoidal, pos_L_max=0)

Build a self attention block for feature size `d`, operating on (d, L, B).
Supported positional encodings:
- `:none`         : no PE
- `:sinusoidal`   : PositionalEmbeddings.AbsolutePE(d, pos_L_max)
- `:learned`      : Flux.Embedding(pos_L_max => d)
`pos_L_max` must be > 0 for `:sinusoidal` or `:learned`.
"""
function TransformerBlock(d::Int;
    nheads::Int,
    ff_mult::Int=4,
    attention_dropout_probability::Real=0.0,
    positional_encoding_kind::Symbol=:sinusoidal,
    pos_L_max::Int=0)

    pos_layer =
        positional_encoding_kind === :none       ? nothing :
        positional_encoding_kind === :sinusoidal ? (pos_L_max > 0 ?
                                                    PositionalEmbeddings.AbsolutePE(d, pos_L_max) :
                                                    error("Provide pos_L_max>0 for :sinusoidal")) :
        positional_encoding_kind === :learned    ? (pos_L_max > 0 ?
                                                    Flux.Embedding(pos_L_max => d) :
                                                    error("Provide pos_L_max>0 for :learned")) :
        error("Unknown positional_encoding_kind=$(positional_encoding_kind)")

    mha = Flux.MultiHeadAttention(d; nheads=nheads, dropout_prob=attention_dropout_probability)
    ln1 = Flux.LayerNorm(d)
    ff  = Flux.Chain(Flux.Dense(d => ff_mult*d, gelu), Flux.Dense(ff_mult*d => d))
    ln2 = Flux.LayerNorm(d)

    return TransformerBlock(positional_encoding_kind, pos_layer, mha, ln1, ff, ln2)
end

# Apply AbsolutePE (library expects (L, d, B)); wrap to/from (d, L, B)
add_absolute_pe(X::AbstractArray, pe_layer) = begin
    X_perm = permutedims(X, (2, 1, 3))  # (L, d, B)
    X_pos  = pe_layer(X_perm)           # (L, d, B)
    permutedims(X_pos, (2, 1, 3))       # (d, L, B)
end

# One forward pass of a TransformerBlock with optional gate (1, L, B)
function forward_block(blk::TransformerBlock, X::AbstractArray; gate::Union{Nothing,AbstractArray}=nothing)
    @assert ndims(X) == 3 "TransformerBlock expects (d, L, B)"
    _, L, B = size(X)

    # Add positional information
    Xp = if blk.positional_encoding_kind === :none || blk.positional_layer === nothing
        X
    elseif blk.positional_encoding_kind === :sinusoidal
        add_absolute_pe(X, blk.positional_layer)
    elseif blk.positional_encoding_kind === :learned
        idx = reshape(collect(1:L), L, 1)
        E   = blk.positional_layer(idx)              # (d, L, 1)
        X .+ E
    else
        error("Unknown positional_encoding_kind=$(blk.positional_encoding_kind)")
    end

    # Gate (mask) after PE to keep PAD silent; gate should be 1 for keep, 0 for pad
    if gate !== nothing
        @assert size(gate) == (1, L, B)
        Xp = Xp .* gate
    end

    # Pre-LN + MHA + residual
    Xn1 = blk.norm_before_attention(Xp)
    attn_out = blk.attention(Xn1)
    A = attn_out isa Tuple ? attn_out[1] : attn_out
    X1 = Xp .+ A

    # Pre-LN + FF + residual
    Xn2 = blk.norm_before_feedforward(X1)
    F   = blk.feedforward(Xn2)
    X1 .+ F
end

# Utilities
mean_over_len(X) = dropdims(mean(X; dims=2), dims=2)
getprop(nt, s::Symbol, default) = (s in propertynames(nt)) ? getproperty(nt, s) : default



"""
Configuration fields expected in `cfg` (NamedTuple or struct):
- d_embed::Int, d_hid::Int, d_out::Int
- num_tokens::Int
- l_heads::Int, l_ff_mult::Int
- h_heads::Int, h_ff_mult::Int
- dropout::Real
- pad_id::Int (set to the actual pad id; use 0 to disable masking)
- (optional) T::Int — typical max length to size positional tables if you do not pass `pos_L_max`
"""
# Initialize (low_state, high_state) as (d_hid, batch) Float32 arrays.
init_states(batch::Int, d_hid::Int) =
    (zeros(Float32, d_hid, batch), zeros(Float32, d_hid, batch))

# ---- Builders ----

"""
    build_models_addcarry(cfg;
        arch::Symbol = :HL,                     # :HL or :HH
        l_positional_encoding_kind::Symbol = :none,
        h_positional_encoding_kind::Symbol = :sinusoidal,
        pos_L_max::Int = 0)

Returns a NamedTuple `models` with fields depending on `arch` and a marker `arch`.
For :HL it builds (Lblock + Hblock). For :HH it builds (H1block + H2block).
"""
function build_models_addcarry(cfg;
    arch::Symbol = :HL,
    l_positional_encoding_kind::Symbol = :none,
    h_positional_encoding_kind::Symbol = :sinusoidal,
    pos_L_max::Int = 0)

    d_embed     = cfg.d_embed
    d_hid       = cfg.d_hid
    d_out       = cfg.d_out
    num_tokens  = cfg.num_tokens
    l_heads     = cfg.l_heads
    l_ff_mult   = cfg.l_ff_mult
    h_heads     = cfg.h_heads
    h_ff_mult   = cfg.h_ff_mult
    dropout     = cfg.dropout

    # Safe default for positional tables
    pos_L_max_eff = pos_L_max > 0 ? pos_L_max : max(getprop(cfg, :T, 0), 3, 512)

    # Input encoder: token embeddings -> hidden
    tok_emb    = num_tokens > 0 ? Flux.Embedding(num_tokens => d_embed) : nothing
    emb_to_hid = Flux.Dense(d_embed => d_hid, gelu)

    # Shared final heads
    post_high = Flux.Chain(Flux.Dense(d_hid => d_hid, gelu))
    fO        = Flux.Chain(Flux.Dense(d_hid => d_out))   # e.g., d_out = 2 for verification

    # Learnable CLS for H with length L+1 path
    h_cls = 0.02f0 .* randn(Float32, d_hid)

    if arch === :HL
        # Adapters for 3-token micro-sequence: (low, task, high)
        l_token_from_low  = Flux.Dense(d_hid => d_hid)
        l_token_from_task = Flux.Dense(d_hid => d_hid)
        l_token_from_high = Flux.Dense(d_hid => d_hid)

        Lblk = TransformerBlock(d_hid;
            nheads = l_heads,
            ff_mult = l_ff_mult,
            attention_dropout_probability = dropout,
            positional_encoding_kind = l_positional_encoding_kind,
            pos_L_max = (l_positional_encoding_kind === :none ? 0 : 3))  # micro-seq length = 3

        Hblk = TransformerBlock(d_hid;
            nheads = h_heads,
            ff_mult = h_ff_mult,
            attention_dropout_probability = dropout,
            positional_encoding_kind = h_positional_encoding_kind,
            pos_L_max = pos_L_max_eff)

        return (; arch, tok_emb, emb_to_hid,
                 l_token_from_low, l_token_from_task, l_token_from_high,
                 Lblk, Hblk, post_high, fO, h_cls)

    elseif arch === :HH
        # Two global blocks (H1 across input tokens, H2 with CLS across H1 outputs)
        H1blk = TransformerBlock(d_hid;
            nheads = l_heads,            # "lower" global block uses L hyperparams
            ff_mult = l_ff_mult,
            attention_dropout_probability = dropout,
            positional_encoding_kind = h_positional_encoding_kind,
            pos_L_max = pos_L_max_eff)

        H2blk = TransformerBlock(d_hid;
            nheads = h_heads,
            ff_mult = h_ff_mult,
            attention_dropout_probability = dropout,
            positional_encoding_kind = h_positional_encoding_kind,
            pos_L_max = pos_L_max_eff+1)

        H1post = Flux.Chain(Flux.Dense(d_hid => d_hid, gelu))
        return (; arch, tok_emb, emb_to_hid, H1blk, H1post, H2blk, post_high, fO, h_cls)
    else
        error("Unknown arch = $(arch). Use :HL or :HH.")
    end
end

# ---- Runners (sequence tokens; verification head) ----

"""
    run_sequence_addcarry!(models, token_ids, low_state, high_state;
                           N::Int, cfg)

- `token_ids`     :: (L, B) Int matrix of token ids
- `low_state`     :: (d_hid, B)
- `high_state`    :: (d_hid, B)
- `N`             :: number of outer HRM cycles
Returns `(yhat, low_state_out, high_state_out)` where `yhat :: (d_out, B)`.
"""
function run_sequence_addcarry!(models,
                                token_ids::AbstractMatrix{<:Integer},
                                low_state::AbstractArray,
                                high_state::AbstractArray;
                                N::Int,
                                cfg)

    if models.arch === :HL
        return _run_sequence_HL!(models, token_ids, low_state, high_state; N=N, cfg=cfg)
    elseif models.arch === :HH
        return _run_sequence_HH!(models, token_ids, low_state, high_state; N=N, cfg=cfg)
    else
        error("Unknown models.arch = $(models.arch)")
    end
end

# ---- Implementation: H+L ----
function _run_sequence_HL!(models,
                           token_ids::AbstractMatrix{<:Integer},
                           low_state::AbstractArray,
                           high_state::AbstractArray;
                           N::Int,
                           cfg)

    @assert cfg.num_tokens > 0 "H+L path expects token IDs"
    L, B = size(token_ids)
    d_hid = size(low_state, 1)

    # Key-padding mask (true = keep); pad_id==0 disables masking
    pad_id   = get(cfg, :pad_id, 0)
    pad_mask = pad_id == 0 ? trues(L, B) : (token_ids .!= pad_id)

    # Embed each token and project to task space stepwise
    # E_t :: (d_embed, B) -> e_t :: (d_hid, B)
    @inbounds for k in 1:N
        Lbuf = Zygote.Buffer(zeros(Float32, d_hid, L, B))

        for t in 1:L
            Et  = models.tok_emb(@view token_ids[t, :])   # (d_embed, B)
            et  = models.emb_to_hid(Et)                   # (d_hid,   B)

            tok_low = models.l_token_from_low(low_state)     # (d_hid, B)
            tok_tsk = models.l_token_from_task(et)           # (d_hid, B)
            tok_hig = models.l_token_from_high(high_state)   # (d_hid, B)

            Xl = cat(
                reshape(tok_low, d_hid, 1, B),
                reshape(tok_tsk, d_hid, 1, B),
                reshape(tok_hig, d_hid, 1, B);
                dims = 2
            )                                                # (d_hid, 3, B)

            Xl_out   = forward_block(models.Lblk, Xl)        # (d_hid, 3, B)
            low_new  = @view Xl_out[:, 1, :]                 # take token-1 as next low

            # Masked functional update across the batch
            keep = reshape(Float32.(pad_mask[t, :]), 1, B)   # (1, B)
            low_state = low_state .* (1 .- keep) .+ low_new .* keep

            Lbuf[:, t, :] = low_state
        end

        # H attends across the collected low states
        H_body = copy(Lbuf)                                  # (d_hid, L, B)
        B_     = size(H_body, 3)
        d_hid_ = size(H_body, 1)

        # Prepend CLS and gate out PAD time steps; keep CLS as 1s
        cls  = repeat(reshape(models.h_cls, d_hid_, 1, 1), 1, 1, B_)  # (d_hid,1,B)
        Hin  = cat(cls, H_body; dims=2)                               # (d_hid,L+1,B)

        W      = reshape(Float32.(pad_mask), 1, L, B_)                # (1, L, B)
        W_ext  = cat(ones(Float32, 1, 1, B_), W; dims=2)              # (1, L+1, B)
        Hin    = Hin .* W_ext

        Hout = forward_block(models.Hblk, Hin; gate=W_ext)
        Hvec = Hout[:, 1, :]                          # CLS
        Hvec = models.post_high(Hvec)
        high_state = high_state .+ Hvec
    end

    yhat = models.fO(high_state)                                 # (d_out, B)
    return yhat, low_state, high_state
end

# ---- Implementation: H+H ----
function _run_sequence_HH!(models,
                           token_ids::AbstractMatrix{<:Integer},
                           low_state::AbstractArray,
                           high_state::AbstractArray;
                           N::Int,
                           cfg)

    @assert cfg.num_tokens > 0 "H+H path expects token IDs"
    L, B = size(token_ids)

    # Mask (true = keep); pad_id==0 disables masking
    pad_id   = get(cfg, :pad_id, 0)
    pad_mask = pad_id == 0 ? trues(L, B) : (token_ids .!= pad_id)

    # Embed whole sequence once
    E = models.tok_emb(token_ids)                 # (d_embed, L, B)

    # Project every step to hidden (task) space
    d_embed = size(E, 1)
    d_hid   = size(high_state, 1)

    Tbuf = Zygote.Buffer(zeros(Float32, d_hid, L, B))
    @inbounds for t in 1:L
        Tbuf[:, t, :] = models.emb_to_hid(E[:, t, :])    # (d_hid, B)
    end
    task_seq = copy(Tbuf)                                 # (d_hid, L, B)

    @inbounds for k in 1:N
        # H1: attend across *entire* token sequence (masked)
        W   = reshape(Float32.(pad_mask), 1, L, B)        # (1, L, B)
        H1o = forward_block(models.H1blk, task_seq; gate=W)  # (d_hid, L, B)
        H1o = models.H1post(H1o)                          # pointwise MLP on features

        # H2: prepend CLS, attend again, read CLS
        B_     = size(H1o, 3)
        d_hid_ = size(H1o, 1)
        cls  = repeat(reshape(models.h_cls, d_hid_, 1, 1), 1, 1, B_)  # (d_hid,1,B)
        Hin  = cat(cls, H1o; dims=2)                                  # (d_hid,L+1,B)
        W2   = cat(ones(Float32, 1, 1, B_), W; dims=2)                # (1, L+1, B)

        H2o = forward_block(models.H2blk, Hin; gate=W2)
        Hvec = H2o[:, 1, :]                                           # CLS
        Hvec = models.post_high(Hvec)
        high_state = high_state .+ Hvec
        # low_state is unused in H+H; keep it as-is for API parity
    end

    yhat = models.fO(high_state)                                      # (d_out, B)
    return yhat, low_state, high_state
end

# ============ Exports ============
export TransformerBlock, forward_block, mean_over_len, init_states,
       build_models_addcarry, run_sequence_addcarry!

end # module
