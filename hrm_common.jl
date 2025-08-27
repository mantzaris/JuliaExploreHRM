
using Lux, Zygote, Random, Optimisers, NNlib, Statistics


module HRMCommon

using Lux, NNlib, Statistics

# tokenwise adapter: run (d,B) layers over (d,L,B)
_as3d(X) = ndims(X) == 2 ? (reshape(X, size(X,1), 1, size(X,2)), true) :
          ndims(X) == 3 ? (X, false) :
          error("Expected 2-D or 3-D tensor, got ndims=$(ndims(X))")

"Apply any (d,B) layer/Chain tokenwise to X::(d,L,B)."
function apply_tokenwise(layer, ps_layer, st_layer, X)
    X3, was2d = _as3d(X)
    d, L, B = size(X3)
    X2 = reshape(X3, d, L * B)
    Y2, st2 = Lux.apply(layer, X2, ps_layer, st_layer)
    Y3 = reshape(Y2, d, L, B)
    return was2d ? dropdims(Y3; dims=2) : Y3, st2
end


function make_positional_layer(d::Int; kind::Symbol = :sinusoidal)
    if kind === :none
        layer   = nothing
        applyfn = (X, _layer, _ps, st) -> (X, st)   # no-op
        return layer, applyfn

    elseif kind === :sinusoidal
        pe = Lux.SinusoidalPositionalEmbedding(d)
        applyfn = (X, layer, ps, st) -> begin
            # ensure we work with (d, L, B)
            X3, _ = _as3d(X)
            L = size(X3, 2)
            B = size(X3, 3)

            # feed (L, B) into the PE layer so output is (d, L, B)
            # content of this dummy input is irrelevant; only its shape matters.
            dummy = zeros(eltype(X3), L, B)
            P, st2 = Lux.apply(layer, dummy, ps, st)  # P :: (d, L, B)
            return X .+ P, st2
        end
        return pe, applyfn

    elseif kind === :learned
        @assert L_max > 0 "Provide L_max for learned positional embeddings"
        embed = Lux.Embedding(L_max => d)
        applyfn = (X, layer, ps, st) -> begin
            X3, _ = _as3d(X)
            L, B = size(X3, 2), size(X3, 3)
            positions = reshape(1:L, L, 1)              # (L, 1)
            E, st2 = Lux.apply(layer, positions, ps, st) # (d, L)
            P = reshape(E, d, L, 1) .* ones(eltype(X3), 1, 1, B)  # (d, L, B)
            return X .+ P, st2
        end
        return embed, applyfn

    else
        error("Unknown positional kind: $kind (use :none, :sinusoidal, or :learned)")
    end
end


"""
make_transformer_block(d;
    nheads,
    ff_mult=4,
    attention_dropout_probability=0.0,
    pos_kind=:sinusoidal,
    pos_L_max::Int=0,
    dense_kwargs=(;))

Returns a NamedTuple:
  (pos_layer, pos_apply, mha, ln1, ff, ln2)

- `pos_layer, pos_apply` implement positional encodings (or no-op).
- Inputs/outputs of the block are (d, L, B).
"""
function make_transformer_block(d::Int;
                                nheads::Int,
                                ff_mult::Int = 4,
                                attention_dropout_probability::Float64 = 0.0,
                                pos_kind::Symbol = :sinusoidal,
                                pos_L_max::Int = 0,         
                                dense_kwargs = (;))
    
    # pos_layer, pos_apply = make_positional_layer(d; kind=pos_kind, L_max=pos_L_max)

    pos_layer, pos_apply = make_positional_layer(d; kind=pos_kind)

    mha = Lux.MultiHeadAttention(d;
        nheads = nheads,
        attention_dropout_probability = attention_dropout_probability)
    ln1 = Lux.LayerNorm(d)
    ff  = Lux.Chain(
            Lux.Dense(d => ff_mult * d, NNlib.gelu; dense_kwargs...),
            Lux.Dense(ff_mult * d => d;            dense_kwargs...)
          )
    ln2 = Lux.LayerNorm(d)
    return (; pos_layer, pos_apply, mha, ln1, ff, ln2)
end


function transformer_forward!(blk, ps_blk, st_blk, X)
    # 0 positional encodings (if any)
    if blk.pos_layer === nothing
        Xp = X
        st_pos = nothing
    else
        Xp, st_pos = blk.pos_apply(X, blk.pos_layer, ps_blk.pos_layer, st_blk.pos_layer)
    end

    # 1 Pre-LN + MHA + residual
    Xn1, st_ln1 = apply_tokenwise(blk.ln1, ps_blk.ln1, st_blk.ln1, Xp)

    # MHA returns ( (A, attn_scores), st_mha )
    (A, _attn_scores), st_mha =
        Lux.apply(blk.mha, (Xn1, Xn1, Xn1), ps_blk.mha, st_blk.mha)

    X1 = Xp .+ A

    # 2) Pre-LN + FFN + residual
    Xn2, st_ln2 = apply_tokenwise(blk.ln2, ps_blk.ln2, st_blk.ln2, X1)
    F,   st_ff  = apply_tokenwise(blk.ff,  ps_blk.ff,  st_blk.ff,  Xn2)
    Y = X1 .+ F

    st_out = (; pos_layer = st_pos, mha=st_mha, ln1=st_ln1, ff=st_ff, ln2=st_ln2)
    return Y, st_out
end



mean_over_len(X) = dropdims(mean(X; dims=2), dims=2)  # (d,L,B)->(d,B)

end # module
