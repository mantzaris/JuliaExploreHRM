module HRMFluxHH

using Flux
using NNlib: gelu
using Zygote
using Random, Statistics
using PositionalEmbeddings 
using Adapt


device_similar(template::AbstractArray, ::Type{T}, dims::Int...) where {T} =
    similar(template, T, dims...)

device_zeros(template::AbstractArray, ::Type{T}, dims::Int...) where {T} =
    fill!(device_similar(template, T, dims...), zero(T))

device_ones(template::AbstractArray, ::Type{T}, dims::Int...) where {T} =
    fill!(device_similar(template, T, dims...), one(T))

adapt_like(x, template) = Adapt.adapt(typeof(template), x)


function __init__()
    try
        @eval using CUDA
        CUDA.allowscalar(false)
    catch
        # CUDA not available; skip
    end
end




# Transformer block (Pre-LN MHA)
struct TransformerBlock{PL}
    positional_encoding_kind::Symbol
    positional_layer::PL
    attention::Flux.MultiHeadAttention
    norm_before_attention::Flux.LayerNorm
    feedforward::Flux.Chain
    norm_before_feedforward::Flux.LayerNorm
end

Flux.@functor TransformerBlock   


"""
    TransformerBlock(d;
        nheads,
        ff_mult=4,
        attention_dropout_probability=0.0,
        positional_encoding_kind=:sinusoidal,
        pos_L_max=0)

Construct a block for feature size `d`. Supported positional encodings:
- `:none`         : no positional encoding
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
    ff  = Flux.Chain(
            Flux.Dense(d => ff_mult*d, gelu),
            Flux.Dense(ff_mult*d => d)
          )
    ln2 = Flux.LayerNorm(d)
    return TransformerBlock(positional_encoding_kind, pos_layer, mha, ln1, ff, ln2)
end




# AbsolutePE expects (L, d, B); we hold (d, L, B)
add_absolute_pe(X::AbstractArray, pe_layer) = begin
    X_perm = permutedims(X, (2, 1, 3))        # (L, d, B)
    X_pos  = pe_layer(X_perm)                  # (L, d, B) (may be CPU)
    X_pos  = adapt_like(X_pos, X)              # ensure same device/type as X
    permutedims(X_pos, (2, 1, 3))              # (d, L, B)
end


"""
forward_block(blk, X; gate=nothing)

- X :: (d, L, B)
- gate :: (1, L, B) multiplier, applied **after** positional encodings to keep PAD truly silent.
"""
function forward_block(blk::TransformerBlock, X::AbstractArray; gate::Union{Nothing,AbstractArray}=nothing)
    @assert ndims(X) == 3 "TransformerBlock expects (d, L, B)"
    d, L, B = size(X)

    Xp = if blk.positional_encoding_kind === :none || blk.positional_layer === nothing
        X
    elseif blk.positional_encoding_kind === :sinusoidal
        add_absolute_pe(X, blk.positional_layer)
    elseif blk.positional_encoding_kind === :learned
        idx = reshape(collect(1:L), L, 1)
        idx = adapt_like(idx, X)                     # indices on same device as X
        E   = blk.positional_layer(idx)              # (d, L, 1)
        X .+ E
    else
        error("Unknown positional_encoding_kind=$(blk.positional_encoding_kind)")
    end

    if gate !== nothing
        @assert size(gate) == (1, L, B)
        # match device *and* dtype (important for mixed precision)
        gate_dev = adapt_like(eltype(X).(gate), X)
        Xp = Xp .* gate_dev
    end

    Xn1 = blk.norm_before_attention(Xp)
    attn_out = blk.attention(Xn1)
    A = attn_out isa Tuple ? attn_out[1] : attn_out
    X1 = Xp .+ A

    Xn2 = blk.norm_before_feedforward(X1)
    F   = blk.feedforward(Xn2)
    return X1 .+ F
end

mean_over_len(X) = dropdims(mean(X; dims=2), dims=2)
make_transformer_block(args...; kwargs...) = TransformerBlock(args...; kwargs...)
getprop(nt, s::Symbol, default) = (s in propertynames(nt)) ? getproperty(nt, s) : default




# H+H model (two "high" modules)
"""
build_models(cfg; l_positional_encoding_kind=:sinusoidal, h_positional_encoding_kind=:sinusoidal, pos_L_max=0)

H+H baseline:
- Two slow modules (H1, H2). Each attends over the **entire input sequence** once per outer cycle.
- No per-token micro-controller (no L).
- For fairness with your cfg, we map:
    H1 uses cfg.l_heads / cfg.l_ff_mult
    H2 uses cfg.h_heads / cfg.h_ff_mult
"""
function build_models(cfg;
    l_positional_encoding_kind::Symbol = :sinusoidal,
    h_positional_encoding_kind::Symbol = :sinusoidal,
    pos_L_max::Int = 0)

    d_in        = cfg.d_in
    d_hid       = cfg.d_hid
    d_out       = cfg.d_out
    d_embed     = cfg.d_embed
    num_tokens  = cfg.num_tokens
    h1_heads    = cfg.l_heads
    h1_ff_mult  = cfg.l_ff_mult
    h2_heads    = cfg.h_heads
    h2_ff_mult  = cfg.h_ff_mult
    dropout     = cfg.dropout

    # safe PE length default
    pos_L_max_eff = pos_L_max > 0 ? pos_L_max : max(getprop(cfg, :T, 0), 3, 512)

    # input encoders
    tok_emb    = num_tokens > 0 ? Flux.Embedding(num_tokens => d_embed) : nothing
    emb_to_hid = Flux.Dense(d_embed => d_hid, gelu)   # IDs path
    raw_to_hid = Flux.Dense(d_in    => d_hid, gelu)   # raw float path (kept for completeness)

    # Two "high" Transformer blocks
    H1blk = TransformerBlock(d_hid;
        nheads = h1_heads,
        ff_mult = h1_ff_mult,
        attention_dropout_probability = dropout,
        positional_encoding_kind = l_positional_encoding_kind,   # map "l" -> H1
        pos_L_max = pos_L_max_eff)

    H2blk = TransformerBlock(d_hid;
        nheads = h2_heads,
        ff_mult = h2_ff_mult,
        attention_dropout_probability = dropout,
        positional_encoding_kind = h_positional_encoding_kind,   # map "h" -> H2
        pos_L_max = pos_L_max_eff)

    # post-CLS heads for H1 and H2, and final readout
    H1post = Flux.Chain(Flux.Dense(d_hid => d_hid, gelu))
    H2post = Flux.Chain(Flux.Dense(d_hid => d_hid, gelu))
    fO     = Flux.Chain(Flux.Dense(d_hid => d_out))

    # learned CLS vectors (small init)
    h1_cls = Flux.param(0.02f0 .* randn(Float32, d_hid))
    h2_cls = Flux.param(0.02f0 .* randn(Float32, d_hid))

    return (; tok_emb, emb_to_hid, raw_to_hid,
             H1blk, H2blk, H1post, H2post, fO,
             h1_cls, h2_cls)
end

"Initialize two slow states (H1, H2) as (d_hid, batch) arrays."
init_states(batch::Int, d_hid::Int; like::Union{Nothing,AbstractArray}=nothing) =
    like === nothing ?
        (zeros(Float32, d_hid, batch), zeros(Float32, d_hid, batch)) :
        (fill!(similar(like, eltype(like), d_hid, batch), zero(eltype(like))),
         fill!(similar(like, eltype(like), d_hid, batch), zero(eltype(like))))



"""
run_sequence_segment!(models, token_ids, high1_state, high2_state; N, cfg)

H+H outer loop:
- Precompute token embeddings -> (d_hid, L, B).
- For each outer cycle k:
    * H1 attends over (CLS1 + e_seq) and updates high1_state once.
    * H2 attends over (CLS2 + H1_body) and updates high2_state once.
- Readout from high2_state.

Returns (yhat, high1_state, high2_state).
"""
function run_sequence_segment!(models,
                               token_ids::AbstractMatrix{<:Integer},
                               high1_state::AbstractArray,
                               high2_state::AbstractArray;
                               N::Int,
                               cfg,
                               lengths::Union{Nothing,AbstractVector{<:Integer}}=nothing)
    @assert cfg.num_tokens > 0 "run_sequence_segment! expects token IDs"
    L, B  = size(token_ids)
    d_hid = size(high1_state, 1)

    # key-padding mask (true = keep)
    pad_id   = getprop(cfg, :pad_id, 0)               # use helper for NamedTuple
    pad_mask = pad_id == 0 ? trues(L, B) : (token_ids .!= pad_id)

    # Precompute per-step task embeddings e_t -> (d_hid, L, B) on *same device/dtype* as states
    Tbuf = Zygote.Buffer(device_zeros(high1_state, eltype(high1_state), d_hid, L, B))
    @inbounds for t in 1:L
        ids_t = @view token_ids[t, :]
        E_t   = models.tok_emb(adapt_like(ids_t, high1_state))  # indices -> device of activations
        e_t   = models.emb_to_hid(E_t)                          # (d_hid, B)
        Tbuf[:, t, :] = e_t
    end
    e_seq = copy(Tbuf)                                          # (d_hid, L, B)

    # CLS and gate tensors match *device & dtype* of e_seq
    B_     = B
    d_hid_ = d_hid

    # CLS vectors (dtype first, then device)
    cls1_cpu = reshape(eltype(e_seq).(models.h1_cls), d_hid_, 1, 1)
    cls2_cpu = reshape(eltype(e_seq).(models.h2_cls), d_hid_, 1, 1)
    cls1     = repeat(adapt_like(cls1_cpu, e_seq), 1, 1, B_)   # (d_hid, 1, B)
    cls2     = repeat(adapt_like(cls2_cpu, e_seq), 1, 1, B_)   # (d_hid, 1, B)

    # Gate: (1, L+1, B), keep CLS as ones
    TH    = eltype(e_seq)
    W_cpu = reshape(TH.(pad_mask), 1, L, B_)                   # (1, L, B)
    W     = adapt_like(W_cpu, e_seq)                           # device match
    W_ext = cat(device_ones(e_seq, TH, 1, 1, B_), W; dims=2)   # (1, L+1, B)

    @inbounds for k in 1:N
        # H1: attends over full input sequence (CLS1 + e_seq); one slow update
        H1_in  = cat(cls1, e_seq; dims=2)                      # (d_hid, L+1, B)
        H1_out = forward_block(models.H1blk, H1_in; gate=W_ext)
        H1_vec = models.H1post(H1_out[:, 1, :])                # read CLS
        high1_state = high1_state .+ H1_vec

        # H2: attends over H1's body (CLS2 + H1_body); one slow update
        H1_body = @view H1_out[:, 2:end, :]                    # (d_hid, L, B)
        H2_in   = cat(cls2, H1_body; dims=2)                   # (d_hid, L+1, B)
        H2_out  = forward_block(models.H2blk, H2_in; gate=W_ext)
        H2_vec  = models.H2post(H2_out[:, 1, :])
        high2_state = high2_state .+ H2_vec
    end

    yhat = models.fO(high2_state)                              # (d_out, B)
    return yhat, high1_state, high2_state
end


export TransformerBlock, forward_block, make_transformer_block, mean_over_len,
       build_models, init_states, run_sequence_segment!

end # module
