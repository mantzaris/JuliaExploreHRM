
using Lux, Zygote, Random, Optimisers, NNlib, Statistics


module HRMCommon

using Lux, NNlib, Random


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

export Tokenwise, TransformerBlock, make_transformer_block, mean_over_len

end # module
