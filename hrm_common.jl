
using Lux, Zygote, Random, Optimisers, NNlib, Statistics


module HRMCommon

using Lux, NNlib, Zygote, Random, Statistics


# minimal wrapper: apply any (d,B) layer tokenwise over (d,L,B)
struct Tokenwise{L}
    layer::L
end

function Lux.setup(rng::AbstractRNG, tw::Tokenwise)
    ps, st = Lux.setup(rng, tw.layer)
    return (; layer=ps), (; layer=st)
end

function Lux.apply(tw::Tokenwise, X, ps, st)
    @assert ndims(X) == 3 "Tokenwise expects (d, L, B); got ndims=$(ndims(X))"
    d, L, B = size(X)
    X2 = reshape(X, d, L*B)
    Y2, st_layer = Lux.apply(tw.layer, X2, ps.layer, st.layer)
    return reshape(Y2, d, L, B), (; layer=st_layer)
end

# single public block: Pre-LN + MHA + residual; Pre-LN + FF + residual
struct TransformerBlock{PL,M,LN1,FF,LN2}
    pos_kind::Symbol
    pos_layer::PL
    mha::M
    ln1::LN1
    ff::FF
    ln2::LN2
end

function TransformerBlock(d::Int;
    nheads::Int,
    ff_mult::Int=4,
    attention_dropout_probability::Real=0.0,
    pos_kind::Symbol=:sinusoidal,
    pos_L_max::Int=0,
    dense_kwargs=(;))

    pos_layer =
        pos_kind === :none        ? nothing :
        pos_kind === :sinusoidal  ? Lux.SinusoidalPositionalEmbedding(d) :
        pos_kind === :learned     ? (pos_L_max > 0 ? Lux.Embedding(pos_L_max => d)
                                                   : error("Provide pos_L_max>0 for :learned")) :
        error("Unknown pos_kind=$pos_kind")

    mha = Lux.MultiHeadAttention(d; nheads=nheads,
        attention_dropout_probability=attention_dropout_probability)

    ln1 = Tokenwise(Lux.LayerNorm(d))
    ff  = Tokenwise(Lux.Chain(
            Lux.Dense(d => ff_mult*d, NNlib.gelu; dense_kwargs...),
            Lux.Dense(ff_mult*d => d;             dense_kwargs...)
          ))
    ln2 = Tokenwise(Lux.LayerNorm(d))

    return TransformerBlock(pos_kind, pos_layer, mha, ln1, ff, ln2)
end

function Lux.setup(rng::AbstractRNG, blk::TransformerBlock)
    ps_pos, st_pos = blk.pos_layer === nothing ? ((;), (;)) : Lux.setup(rng, blk.pos_layer)
    ps_mha, st_mha = Lux.setup(rng, blk.mha)
    ps_ln1, st_ln1 = Lux.setup(rng, blk.ln1)
    ps_ff,  st_ff  = Lux.setup(rng, blk.ff)
    ps_ln2, st_ln2 = Lux.setup(rng, blk.ln2)
    return (; pos=ps_pos, mha=ps_mha, ln1=ps_ln1, ff=ps_ff, ln2=ps_ln2),
           (; pos=st_pos, mha=st_mha, ln1=st_ln1, ff=st_ff, ln2=st_ln2)
end

function Lux.apply(blk::TransformerBlock, X, ps, st)
    @assert ndims(X) == 3 "TransformerBlock expects (d, L, B)"
    d, L, B = size(X)

    # positional encoding 
    if blk.pos_kind === :none
        Xp = X
        st_pos = st.pos
    elseif blk.pos_kind === :sinusoidal
        P, stp = Lux.apply(blk.pos_layer, zeros(eltype(X), L, B), ps.pos, st.pos)  # (d,L,B)
        Xp, st_pos = X .+ P, stp
    elseif blk.pos_kind === :learned
        E, stp = Lux.apply(blk.pos_layer, reshape(1:L, L, 1), ps.pos, st.pos)     # (d,L)
        Xp, st_pos = X .+ reshape(E, d, L, 1), stp                                 # broadcasts over B
    else
        error("Unknown pos_kind=$(blk.pos_kind)")
    end

    # pre-LN + MHA + residual
    Xn1, st_ln1 = Lux.apply(blk.ln1, Xp, ps.ln1, st.ln1)
    (A, _attn), st_mha = Lux.apply(blk.mha, (Xn1, Xn1, Xn1), ps.mha, st.mha)
    X1 = Xp .+ A

    # pre-LN + FF + residual
    Xn2, st_ln2 = Lux.apply(blk.ln2, X1, ps.ln2, st.ln2)
    F,   st_ff  = Lux.apply(blk.ff,  Xn2, ps.ff,  st.ff)
    Y = X1 .+ F

    return Y, (; pos=st_pos, mha=st_mha, ln1=st_ln1, ff=st_ff, ln2=st_ln2)
end

# small helper kept for convenience
mean_over_len(X) = dropdims(mean(X; dims=2), dims=2)

make_transformer_block(args...; kwargs...) = TransformerBlock(args...; kwargs...)


"""
    setup_params_states(models; rng = Random.default_rng())

Convenience wrapper that calls `Lux.setup` for each sub-module in `models` and
returns `(ps, st)` as NamedTuples with the same field names you already use.

- If `models.tok_emb === nothing`, the corresponding `(ps, st)` entries are `nothing`.
- Designed to match your current training loop's expectations exactly.
"""
function setup_params_states(models; rng::AbstractRNG = Random.default_rng())
    # input
    if models.tok_emb === nothing
        psEmb, stEmb = nothing, nothing
    else
        psEmb, stEmb = Lux.setup(rng, models.tok_emb)
    end
    psEH,  stEH  = Lux.setup(rng, models.emb_to_hid)
    psRaw, stRaw = Lux.setup(rng, models.raw_to_hid)

    # L micro‑sequence projections
    psLlow, stLlow = Lux.setup(rng, models.l_token_from_low)
    psLtsk, stLtsk = Lux.setup(rng, models.l_token_from_task)
    psLhig, stLhig = Lux.setup(rng, models.l_token_from_high)

    # transformer blocks (single setup each)
    psL, stL = Lux.setup(rng, models.Lblk)
    psH, stH = Lux.setup(rng, models.Hblk)

    # H post
    psHP, stHP = Lux.setup(rng, models.Hpost)

    # head
    psO,  stO  = Lux.setup(rng, models.fO)

    ps = (Emb=psEmb, EH=psEH, Raw=psRaw,
          Llow=psLlow, Ltsk=psLtsk, Lhig=psLhig,
          L=psL, H=psH, HP=psHP, O=psO)

    st = (Emb=stEmb, EH=stEH, Raw=stRaw,
          Llow=stLlow, Ltsk=stLtsk, Lhig=stLhig,
          L=stL, H=stH, HP=stHP, O=stO)

    return ps, st
end



"""
    build_models(cfg; pos_kind=:sinusoidal)

Construct the small HRM stack using the provided configuration `cfg` (NamedTuple).
Expected `cfg` fields:
  d_in, d_hid, d_out, d_embed, num_tokens,
  l_heads, l_ff_mult, h_heads, h_ff_mult, dropout

Returns a NamedTuple with fields:
  tok_emb, emb_to_hid, raw_to_hid,
  l_token_from_low, l_token_from_task, l_token_from_high,
  Lblk, Hblk, Hpost, fO
"""
function build_models(cfg; pos_kind::Symbol = :sinusoidal)
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

    # input encoders
    tok_emb    = num_tokens > 0 ? Lux.Embedding(num_tokens => d_embed) : nothing
    emb_to_hid = Lux.Dense(d_embed => d_hid, NNlib.gelu)   # IDs path
    raw_to_hid = Lux.Dense(d_in    => d_hid, NNlib.gelu)   # raw float path

    # adapters to form 3-token micro-sequence for the L block
    l_token_from_low  = Lux.Dense(d_hid => d_hid)
    l_token_from_task = Lux.Dense(d_hid => d_hid)
    l_token_from_high = Lux.Dense(d_hid => d_hid)

    # L and H Transformer blocks (use your lean Tokenwise+TransformerBlock)
    Lblk = TransformerBlock(d_hid;
        nheads = l_heads,
        ff_mult = l_ff_mult,
        attention_dropout_probability = dropout,
        pos_kind = pos_kind)

    Hblk = TransformerBlock(d_hid;
        nheads = h_heads,
        ff_mult = h_ff_mult,
        attention_dropout_probability = dropout,
        pos_kind = pos_kind)

    # post-pool head for H, and final readout
    Hpost = Lux.Chain(Lux.Dense(d_hid => d_hid, NNlib.gelu))
    fO    = Lux.Chain(Lux.Dense(d_hid => d_out))

    return (; tok_emb, emb_to_hid, raw_to_hid,
             l_token_from_low, l_token_from_task, l_token_from_high,
             Lblk, Hblk, Hpost, fO)
end



"Initialize (low_state, high_state) as (d_hid, batch) Float32 arrays."
init_states(batch::Int, d_hid::Int) =
    (zeros(Float32, d_hid, batch), zeros(Float32, d_hid, batch))

"""
    run_segment!(models, ps, st, x_in, low_state, high_state; N::Int, T::Int, cfg)

One HRM segment:
  1) Encode input `x_in` into a task vector `e_task` in hidden space.
  2) For `k=1..N` cycles:
     - For each `t=1..T`, form a 3-token micro-sequence (low/task/high) and run the L block.
     - Save the updated low state; after T steps, run the H block over the low-state sequence.
     - Pool H, project with `Hpost`, and residual-update the high state.
  3) Predict with the readout head.

Inputs:
  - `models`: NamedTuple returned by `build_models`
  - `ps, st`: parameter and state trees from `setup_params_states`
  - `x_in`: either raw floats `(d_in, B)` or Int IDs `(d_in, B)` (when `cfg.num_tokens > 0`)
  - `low_state, high_state`: both `(d_hid, B)` Float32
  - Keyword args: `N`, `T`, and `cfg` (your config NamedTuple)

Returns: `(yhat, st_new, low_state_out, high_state_out)`
"""
function run_segment!(models, ps, st, x_in, low_state, high_state; N::Int, T::Int, cfg)
    stEmb, stEH, stRaw = st.Emb, st.EH, st.Raw
    stLlow, stLtsk, stLhig = st.Llow, st.Ltsk, st.Lhig
    stL, stH, stHP = st.L, st.H, st.HP

    # 1) Encode input into task vector e_task (no attention here)
    if cfg.num_tokens > 0
        # x_in are Int IDs of shape (d_in, B)
        E, stEmb = Lux.apply(models.tok_emb, x_in, ps.Emb, st.Emb)                       # (d_embed, d_in, B)
        e_task, stEH = Lux.apply(models.emb_to_hid, dropdims(mean(E; dims=2), dims=2),    # (d_hid, B)
                                 ps.EH, st.EH)
        stRaw = st.Raw
    else
        # raw float path: x_in :: (d_in, B) -> (d_hid, B)
        e_task, stRaw = Lux.apply(models.raw_to_hid, x_in, ps.Raw, st.Raw)               # (d_hid, B)
    end

    # 2) HRM cycles
    @inbounds for k in 1:N
        # Collect low states across T without illegal mutation (Zygote-friendly)
        Lbuf = Zygote.Buffer(zeros(Float32, size(low_state,1), T, size(low_state,2)))    # (d_hid, T, B)

        for t in 1:T
            # 2a) Build 3-token micro-sequence (low, task, high)
            tok_low, stLlow = Lux.apply(models.l_token_from_low,  low_state,  ps.Llow, stLlow)  # (d_hid,B)
            tok_tsk, stLtsk = Lux.apply(models.l_token_from_task, e_task,     ps.Ltsk, stLtsk)  # (d_hid,B)
            tok_hig, stLhig = Lux.apply(models.l_token_from_high, high_state, ps.Lhig, stLhig)  # (d_hid,B)

            # (d_hid, L=3, B)
            Xl = permutedims(cat(tok_low, tok_tsk, tok_hig; dims=3), (1, 3, 2))

            # 2b) Run L block; take token 1 as the updated low state
            Xl_out, stL = Lux.apply(models.Lblk, Xl, ps.L, stL)                           # (d_hid, 3, B)
            @views low_state = Xl_out[:, 1, :]                                            # (d_hid, B)

            # Save low state into buffer
            @views Lbuf[:, t, :] = low_state
        end

        # 2c) Run H block over the T collected low states
        H_in  = copy(Lbuf)                                                                # (d_hid, T, B)
        H_out, stH = Lux.apply(models.Hblk, H_in, ps.H, stH)

        # Pool H over time, project, and residual-update high state
        H_vec = dropdims(mean(H_out; dims=2), dims=2)                                     # (d_hid, B)
        H_vec, stHP = Lux.apply(models.Hpost, H_vec, ps.HP, stHP)
        high_state = high_state .+ H_vec
    end

    # 3) Readout
    yhat, stO = Lux.apply(models.fO, high_state, ps.O, st.O)

    st_new = (Emb=stEmb, EH=stEH, Raw=stRaw,
              Llow=stLlow, Ltsk=stLtsk, Lhig=stLhig,
              L=stL, H=stH, HP=stHP, O=stO)

    return yhat, st_new, low_state, high_state
end




export Tokenwise, TransformerBlock, make_transformer_block, mean_over_len, setup_params_states
export build_models


end # module
