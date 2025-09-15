



module HRMFlux

using Flux
using NNlib: gelu
using Zygote
using Random, Statistics
using PositionalEmbeddings  
using CUDA  # GPU support
using Functors: fmap, @functor
using Adapt 
using CUDA: CuArray, cu
using ChainRulesCore: ignore_derivatives


device_similar(template::AbstractArray, ::Type{T}, dims::Int...) where {T} =
    similar(template, T, dims...)

device_zeros(template::AbstractArray, ::Type{T}, dims::Int...) where {T} =
    ignore_derivatives() do
        template isa CuArray ? CUDA.zeros(T, dims...) : zeros(T, dims...)
    end

device_ones(template::AbstractArray, ::Type{T}, dims::Int...) where {T} =
    ignore_derivatives() do
        template isa CuArray ? CUDA.ones(T,  dims...) : ones(T,  dims...)
    end

Zygote.@nograd device_similar
Zygote.@nograd device_zeros
Zygote.@nograd device_ones



# Adapt 'x' to live on the same device/type as 'template'
adapt_like(x, template) = Adapt.adapt(typeof(template), x)

to_device_like_indices(x, template) = template isa CuArray ? cu(x) : x
Zygote.@nograd to_device_like_indices


# transformer block (Pre-LN, MHA, FF, residual)
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

@functor TransformerBlock


"""
    TransformerBlock(d;
        nheads,
        ff_mult=4,
        attention_dropout_probability=0.0,
        positional_encoding_kind=:sinusoidal,
        pos_L_max=0)

Construct a block for feature size `d`. Supported positional encodings:
- `:none`         : no positional encoding
- `:sinusoidal`   : PositionalEmbeddings.AbsolutePE(d, pos_L_max)  (library)
- `:learned`      : Flux.Embedding(pos_L_max => d)                 (trainable)
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

    # Flux.MultiHeadAttention expects (d_model, seq_len, batch)
    mha = Flux.MultiHeadAttention(d; nheads=nheads, dropout_prob=attention_dropout_probability)

    # LayerNorm(d) normalizes over the feature dim (first dim in (d,L,B))
    ln1 = Flux.LayerNorm(d)
    ff  = Flux.Chain(
            Flux.Dense(d => ff_mult*d, gelu),
            Flux.Dense(ff_mult*d => d)
          )
    ln2 = Flux.LayerNorm(d)

    return TransformerBlock(positional_encoding_kind, pos_layer, mha, ln1, ff, ln2)
end


# AbsolutePE (seq_len, channels, batch) = (L, d, B).
# wrapper adds PE by permuting around the call (no in-place mutation)
add_absolute_pe(X::AbstractArray, pe_layer) = begin
    X_perm = permutedims(X, (2, 1, 3))             # (L, d, B)
    X_pos  = pe_layer(X_perm)                       # may be CPU or GPU depending on layer
    X_pos  = adapt_like(X_pos, X)                   # ensure same device/type as X
    permutedims(X_pos, (2, 1, 3))                   # (d, L, B)
end


function forward_block(blk::TransformerBlock, X::AbstractArray; gate::Union{Nothing,AbstractArray}=nothing)
    @assert ndims(X) == 3 "TransformerBlock expects (d, L, B)"
    d, L, B = size(X)

    # Positional encoding
    Xp = if blk.positional_encoding_kind === :none || blk.positional_layer === nothing
        X
    elseif blk.positional_encoding_kind === :sinusoidal
        add_absolute_pe(X, blk.positional_layer)
    elseif blk.positional_encoding_kind === :learned
        idx = reshape(collect(1:L), L, 1)                # Int indices
        idx_dev = to_device_like_indices(idx, X)         # move to device, keep Int
        E   = blk.positional_layer(idx_dev)              # (d, L, 1)
        X .+ E
    
    else
        error("Unknown positional_encoding_kind=$(blk.positional_encoding_kind)")
    end

    # NEW: gate after PE to keep PAD truly silent (CLS stays 1s in gate)
    if gate !== nothing
        @assert size(gate) == (1, L, B)
        Xp = Xp .* gate
    end

    # Pre-LN + self-attention + residual
    Xn1 = blk.norm_before_attention(Xp)
    attn_out = blk.attention(Xn1)
    A = attn_out isa Tuple ? attn_out[1] : attn_out
    X1 = Xp .+ A

    # Pre-LN + FF + residual
    Xn2 = blk.norm_before_feedforward(X1)
    F   = blk.feedforward(Xn2)
    return X1 .+ F
end



# mean over length dimension (2nd dim)
mean_over_len(X) = dropdims(mean(X; dims=2), dims=2)

make_transformer_block(args...; kwargs...) = TransformerBlock(args...; kwargs...)

getprop(nt, s::Symbol, default) = (s in propertynames(nt)) ? getproperty(nt, s) : default


# build the HRM stack (Flux layers)
function build_models(cfg;
                l_positional_encoding_kind::Symbol = :none,       # changed default
                h_positional_encoding_kind::Symbol = :sinusoidal,
                pos_L_max::Int = 0)    
                
    d_in        = cfg.d_in
    d_hid       = cfg.d_hid
    d_out       = cfg.d_out
    d_embed     = cfg.d_embed
    num_tokens  = cfg.num_tokens
    l_heads     = cfg.l_heads
    l_ff_mult   = cfg.l_ff_mult
    h_heads     = cfg.h_heads
    h_ff_mult   = cfg.h_ff_mult
    dropout     = cfg.dropout

    # pick a safe default for PE lengths if none provided
    pos_L_max_eff = pos_L_max > 0 ? pos_L_max : max(getprop(cfg, :T, 0), 3, 512)

    # input encoders
    tok_emb    = num_tokens > 0 ? Flux.Embedding(num_tokens => d_embed) : nothing
    emb_to_hid = Flux.Dense(d_embed => d_hid, gelu)   # IDs path
    raw_to_hid = Flux.Dense(d_in    => d_hid, gelu)   # raw float path

    # adapters to form 3-token micro-sequence for the L block
    l_token_from_low  = Flux.Dense(d_hid => d_hid)
    l_token_from_task = Flux.Dense(d_hid => d_hid)
    l_token_from_high = Flux.Dense(d_hid => d_hid)

    # L and H Transformer blocks  
    Lblk = TransformerBlock(d_hid;
        nheads = l_heads,
        ff_mult = l_ff_mult,
        attention_dropout_probability = dropout,
        positional_encoding_kind = l_positional_encoding_kind,
        pos_L_max = (l_positional_encoding_kind === :none ? 0 : pos_L_max_eff))

    Hblk = TransformerBlock(d_hid;
        nheads = h_heads,
        ff_mult = h_ff_mult,
        attention_dropout_probability = dropout,
        positional_encoding_kind = h_positional_encoding_kind,
        pos_L_max = pos_L_max_eff)

    
    Hpost = Flux.Chain(Flux.Dense(d_hid => d_hid, gelu))
    fO    = Flux.Chain(Flux.Dense(d_hid => d_out))

    # NEW: learnable CLS vector (small init); trained automatically
    h_cls = 0.02f0 .* randn(Float32, d_hid)

    return (; tok_emb, emb_to_hid, raw_to_hid,
             l_token_from_low, l_token_from_task, l_token_from_high,
             Lblk, Hblk, Hpost, fO,
             h_cls) 
end



# state init and one HRM segment
"Initialize (low_state, high_state) as (d_hid, batch) Float32 arrays."
init_states(batch::Int, d_hid::Int) =
    (zeros(Float32, d_hid, batch), zeros(Float32, d_hid, batch))


"""
    run_segment!(models, x_in, low_state, high_state; N, T, cfg)

Flux version of a single HRM segment. Returns (yhat, low_state_out, high_state_out).
"""
function run_segment!(models, x_in, low_state, high_state; N::Int, T::Int, cfg)
    # 1 encode input into task vector e_task (no attention here)
    if cfg.num_tokens > 0
        # x_in are Int IDs of shape (d_in, B)
        xdev = to_device_like_indices(x_in, low_state)   # keep Int, move to GPU if needed
        E    = models.tok_emb(xdev)
                                 # (d_embed, d_in, B)
        e_task = models.emb_to_hid(dropdims(mean(E; dims=2), dims=2))    # (d_hid, B)
    else
        # raw float path: x_in :: (d_in, B) -> (d_hid, B)
        e_task = models.raw_to_hid(x_in)                                 # (d_hid, B)
    end

    # 2 HRM cycles
    @inbounds for k in 1:N
        # buffer to collect low states across inner steps
        Lbuf = Zygote.Buffer(device_zeros(low_state, eltype(low_state),
                                  size(low_state,1), T, size(low_state,2)))
    
        d_hid = size(low_state, 1)
        B     = size(low_state, 2)


        for t in 1:T
            # build 3-token micro-sequence (low, task, high), (d_hid, 3, B)
            tok_low  = models.l_token_from_low(  low_state)   # (d_hid, B)
            tok_tsk  = models.l_token_from_task(e_task)       # (d_hid, B)
            tok_hig  = models.l_token_from_high(high_state)   # (d_hid, B)

            Xl = cat(
                reshape(tok_low, d_hid, 1, B),
                reshape(tok_tsk, d_hid, 1, B),
                reshape(tok_hig, d_hid, 1, B);
                dims = 2,
            )

            # L block; take token 1 as the updated low state
            Xl_out = forward_block(models.Lblk, Xl)           # (d_hid, 3, B)
            low_state = copy(Xl_out[:, 1, :])   # materialize a standalone tensor

            # Save into buffer (write directly to Buffer, not to a SubArray)
            Lbuf[:, t, :] = low_state
        end

        # H block over collected low states
        H_in  = copy(Lbuf)                                    # (d_hid, T, B)
        H_out = forward_block(models.Hblk, H_in)
        H_vec = dropdims(mean(H_out; dims=2), dims=2)         # (d_hid, B)
        H_vec = models.Hpost(H_vec)                           # (d_hid, B)
        high_state = high_state .+ H_vec
    end

    # 3 readout
    yhat = models.fO(high_state)                              # (d_out, B)

    return yhat, low_state, high_state
end


function build_models_GRU(cfg; positional_encoding_kind::Symbol = :sinusoidal, pos_L_max::Int = 0)
    d_in        = cfg.d_in
    d_hid       = cfg.d_hid
    d_out       = cfg.d_out
    d_embed     = cfg.d_embed
    num_tokens  = cfg.num_tokens
    l_heads     = cfg.l_heads
    l_ff_mult   = cfg.l_ff_mult
    h_heads     = cfg.h_heads
    h_ff_mult   = cfg.h_ff_mult
    dropout     = cfg.dropout

    # pick a safe default for PE lengths if none provided
    pos_L_max_eff = pos_L_max > 0 ? pos_L_max : max(getprop(cfg, :T, 0), 3, 512)

    # input encoders
    tok_emb    = num_tokens > 0 ? Flux.Embedding(num_tokens => d_embed) : nothing
    # tok_rnn    = num_tokens > 0 ? Flux.GRU(d_embed) : nothing   # NEW: sequence encoder
    tok_rnn = nothing

    if num_tokens > 0
        tok_rnn = make_recur_gru(d_embed, d_embed)
    end

    emb_to_hid = Flux.Dense(d_embed => d_hid, gelu)                      # map encoder state -> d_hid
    raw_to_hid = Flux.Dense(d_in    => d_hid, gelu)                      # raw float path (unchanged)

    # adapters to form 3-token micro-sequence for the L block
    l_token_from_low  = Flux.Dense(d_hid => d_hid)
    l_token_from_task = Flux.Dense(d_hid => d_hid)
    l_token_from_high = Flux.Dense(d_hid => d_hid)

    # L and H Transformer blocks (same PE config)
    Lblk = TransformerBlock(d_hid;
        nheads = l_heads,
        ff_mult = l_ff_mult,
        attention_dropout_probability = dropout,
        positional_encoding_kind = positional_encoding_kind,
        pos_L_max = pos_L_max_eff)

    Hblk = TransformerBlock(d_hid;
        nheads = h_heads,
        ff_mult = h_ff_mult,
        attention_dropout_probability = dropout,
        positional_encoding_kind = positional_encoding_kind,
        pos_L_max = pos_L_max_eff)

    # post-pool head for H, and final readout
    Hpost = Flux.Chain(Flux.Dense(d_hid => d_hid, gelu))
    fO    = Flux.Chain(Flux.Dense(d_hid => d_out))
    
    # learnable CLS vector (moved to GPU when you move the model)
    h_cls = 0.02f0 .* randn(Float32, d_hid)
    
    return (; tok_emb, tok_rnn, emb_to_hid, raw_to_hid,
             l_token_from_low, l_token_from_task, l_token_from_high,
             Lblk, Hblk, Hpost, fO, h_cls)
end


function run_segment_GRU!(models, x_in, low_state, high_state; N::Int, T::Int, cfg)
    @assert cfg.num_tokens > 0 "GRU path expects token IDs"

    # Embed and run GRU to get per-step hidden states (use Buffers; no in-place on arrays)
    xdev = to_device_like_indices(x_in, low_state)
    E    = models.tok_emb(xdev)
    
    
    Flux.reset!(models.tok_rnn)

    d_embed, L, B = size(E)
    d_hid = size(low_state, 1)

    Hbuf = Zygote.Buffer(device_zeros(E, eltype(E), d_embed, L, B))
    @inbounds for t in 1:L
        Hbuf[:, t, :] = models.tok_rnn(E[:, t, :])       # write into Buffer (AD-safe)
    end
    H_seq = copy(Hbuf)                                    # materialize once

    # Project each step to task space
    Tbuf = Zygote.Buffer(device_zeros(low_state, eltype(low_state), cfg.d_hid, L, B))
    @inbounds for t in 1:L
        Tbuf[:, t, :] = models.emb_to_hid(H_seq[:, t, :])
    end
    e_task_seq = copy(Tbuf)

    # Mask: true for real tokens, false for PAD
    pad_mask = x_in .!= cfg.pad_id        # (L, B) Bool

    # HRM outer cycles
    @inbounds for k in 1:N
        # collect low states across *all* time steps
        Lbuf = Zygote.Buffer(device_zeros(low_state, eltype(low_state), d_hid, L, B))

        for t in 1:L
            # 3-token micro-sequence for this step
            tok_low = models.l_token_from_low(low_state)        # (d_hid, B)
            tok_tsk = models.l_token_from_task(e_task_seq[:, t, :])  # (d_hid, B)
            tok_hig = models.l_token_from_high(high_state)      # (d_hid, B)

            # L block; propose new low state for all columns
            # Xl_out  = forward_block(models.Lblk, Xl)            # (d_hid, 3, B)
            Xl = cat(
                reshape(tok_low, d_hid, 1, B),
                reshape(tok_tsk, d_hid, 1, B),
                reshape(tok_hig, d_hid, 1, B);
                dims = 2,
            )

            Xl_out = forward_block(models.Lblk, Xl)


            low_new = Xl_out[:, 1, :]

            # Functional masked update (no view writes):
            # for columns where pad_mask[t,b] == true, take low_new[:,b]; else keep low_state[:,b]
            Telt = eltype(low_state)
            Mf_cpu = reshape(Telt.(pad_mask[t, :]), 1, B)
            
            Mf     = adapt_like(Mf_cpu, low_state)
            low_state = low_state .* (1 .- Mf) .+ low_new .* Mf

            Lbuf[:, t, :] = low_state
        end

        # H attends over the whole history; ignore PAD positions in the mean
        H_body = copy(Lbuf)                                      # (d_hid, L, B)
        B_     = size(H_body, 3)
        d_hid_ = size(H_body, 1)

        cls_cpu = reshape(eltype(H_body).(models.h_cls), d_hid_, 1, 1)  # convert dtype first
        cls1    = adapt_like(cls_cpu, H_body)                           # then move to the right device
        cls     = repeat(cls1, 1, 1, B_)
        H_in = cat(cls, H_body; dims=2)                          # (d_hid, L+1, B)

        
        TH = eltype(H_body)
        W_cpu = reshape(TH.(pad_mask), 1, L, B_)
        W     = adapt_like(W_cpu, H_body)
        W_ext = cat(device_ones(H_body, TH, 1, 1, B_), W; dims=2)
        H_in   = H_in .* W_ext
        H_out  = forward_block(models.Hblk, H_in; gate=W_ext)

        H_vec = H_out[:, 1, :]                                # read CLS
        H_vec = models.Hpost(H_vec)
        high_state = high_state .+ H_vec
    end

    yhat = models.fO(high_state)                                # (d_out, B)
    return yhat, low_state, high_state
end



make_recur_gru(din::Int, dhid::Int) = begin
    try
        return Flux.GRU(din => dhid)                 # Pair API
    catch
        return Flux.Recur(Flux.GRUCell(din => dhid)) # Fallback
    end
end


function run_sequence_segment!(models,
                               token_ids::AbstractMatrix{<:Integer},
                               low_state::AbstractArray,
                               high_state::AbstractArray;
                               N::Int,
                               cfg,
                               lengths::Union{Nothing,AbstractVector{<:Integer}}=nothing)
    @assert cfg.num_tokens > 0 "run_sequence_segment! expects token IDs"
    L, B = size(token_ids)
    d_hid = size(low_state, 1)

    # key-padding mask (true = keep)
    pad_id   = getprop(cfg, :pad_id, 0)
    pad_mask = pad_id == 0 ? trues(L, B) : (token_ids .!= pad_id)

    # preallocate 3-token micro-sequence

    @inbounds for k in 1:N
        Lbuf = Zygote.Buffer(device_zeros(low_state, eltype(low_state), d_hid, L, B))

        for t in 1:L
            ids_t = @view token_ids[t, :]
            E_t   = models.tok_emb(to_device_like_indices(ids_t, low_state))


            e_t   = models.emb_to_hid(E_t)                 # (d_hid,   B)

            tok_low = models.l_token_from_low(low_state)
            tok_tsk = models.l_token_from_task(e_t)
            tok_hig = models.l_token_from_high(high_state)

            Xl = cat(
                reshape(tok_low, d_hid, 1, B),
                reshape(tok_tsk, d_hid, 1, B),
                reshape(tok_hig, d_hid, 1, B);
                dims = 2,
            )

            Xl_out = forward_block(models.Lblk, Xl)

            
            low_new = @view Xl_out[:, 1, :]

            # masked low_state update across batch
            Telt = eltype(low_state)
            Mf_cpu = reshape(Telt.(pad_mask[t, :]), 1, B)
            
            Mf     = adapt_like(Mf_cpu, low_state)
            low_state = low_state .* (1 .- Mf) .+ low_new .* Mf

            Lbuf[:, t, :] = low_state
        end

        H_body = copy(Lbuf)                                  # (d_hid, L, B)
        B_     = size(H_body, 3)
        d_hid_ = size(H_body, 1)

        # NEW: prepend CLS along the length axis
        cls_cpu = reshape(eltype(H_body).(models.h_cls), d_hid_, 1, 1)  # convert dtype first
        cls1    = adapt_like(cls_cpu, H_body)                           # then move to the right device
        cls     = repeat(cls1, 1, 1, B_)

        H_in = cat(cls, H_body; dims=2)                               # (d_hid, L+1, B)

        # Gate out PAD time steps BEFORE attention (CLS kept as 1s)
        
        TH = eltype(H_body)
        W_cpu = reshape(TH.(pad_mask), 1, L, B_)
        W     = adapt_like(W_cpu, H_body)
        W_ext = cat(device_ones(H_body, TH, 1, 1, B_), W; dims=2)
        H_in   = H_in .* W_ext

        # H transformer; read CLS output
        H_out  = forward_block(models.Hblk, H_in; gate=W_ext)
        H_vec  = H_out[:, 1, :]

        H_vec      = models.Hpost(H_vec)
        high_state = high_state .+ H_vec
    end

    yhat = models.fO(high_state)
    return yhat, low_state, high_state
end

export TransformerBlock, forward_block, make_transformer_block, mean_over_len,
       build_models, build_models_GRU, init_states,
       run_segment!, run_segment_GRU!, run_sequence_segment!


function __init__()
    try
        CUDA.allowscalar(false)
    catch
        # CUDA not present; ignore
    end
end

    
end # module
