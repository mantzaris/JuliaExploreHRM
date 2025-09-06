module Dyck1Data
using Random

export balanced_string, corrupt, make_dataset,
       is_balanced, maximum_nesting_depth,
       corrupt_strict, make_dataset_strict, swap_one_pair,
       tokenize_parens, batch_tokenize, pad_sequences, make_minibatches, make_splits, make_mask

# ---------------------------
# Core Dyck-1 data generator
# ---------------------------

"Generate a balanced parentheses string with exact length 2n (non-uniform over Catalan)."
function balanced_string(rng::AbstractRNG, n::Int)::String
    s = IOBuffer()
    open_count = 0
    close_count = 0
    total = 2n  # ok in Julia; write 2*n if you prefer explicitness
    for i in 1:total
        remaining  = total - i + 1
        must_open  = (open_count == close_count)                 # cannot close first
        must_close = (open_count - close_count) == remaining     # must close to finish
        if must_open
            print(s, '('); open_count += 1
        elseif must_close
            print(s, ')'); close_count += 1
        else
            if rand(rng) < 0.5
                print(s, '('); open_count += 1
            else
                if open_count > close_count
                    print(s, ')'); close_count += 1
                else
                    print(s, '('); open_count += 1
                end
            end
        end
    end
    return String(take!(s))
end

"Corrupt a balanced string by: drop one char, flip one char, or swap adjacent (may leave some negatives balanced)."
function corrupt(rng::AbstractRNG, s::String)::String
    if isempty(s); return ")"; end
    mode = rand(rng, 1:3)
    if mode == 1
        idx = rand(rng, 1:length(s))
        return s[1:idx-1] * (idx < length(s) ? s[idx+1:end] : "")
    elseif mode == 2
        idx = rand(rng, 1:length(s))
        c = s[idx]
        flip = c == '(' ? ')' : '('
        return s[1:idx-1] * string(flip) * (idx < length(s) ? s[idx+1:end] : "")
    else
        if length(s) ≥ 2
            idx = rand(rng, 1:length(s)-1)
            return s[1:idx-1] * s[idx+1] * s[idx] * (idx+1 < length(s) ? s[idx+2:end] : "")
        else
            return s
        end
    end
end

"Standard dataset (label=1 for balanced, 0 for unbalanced). Negatives may have tiny label-noise."
function make_dataset(rng::AbstractRNG; N::Int=10_000, n_range::UnitRange{Int}=2:8, negative_ratio::Float64=0.5)
    xs = Vector{String}(undef, N)
    ys = Vector{Int}(undef, N)
    for i in 1:N
        is_positive = rand(rng) > negative_ratio   # positives ≈ 1 - negative_ratio
        n = rand(rng, n_range)
        if is_positive
            s = balanced_string(rng, n)
            xs[i] = s; ys[i] = 1
        else
            s = balanced_string(rng, n)
            xs[i] = corrupt(rng, s); ys[i] = 0
        end
    end
    return xs, ys
end

"True checker: returns true iff s is a well-formed Dyck-1 string."
function is_balanced(s::AbstractString)::Bool
    depth = 0
    @inbounds for c in s
        if c == '('
            depth += 1
        elseif c == ')'
            depth -= 1
            if depth < 0
                return false
            end
        else
            return false
        end
    end
    return depth == 0
end

"Maximum nesting depth; returns -1 if not well-formed."
function maximum_nesting_depth(s::AbstractString)::Int
    depth = 0
    maxd  = 0
    @inbounds for c in s
        if c == '('
            depth += 1
            maxd = max(maxd, depth)
        elseif c == ')'
            depth -= 1
        else
            return -1
        end
    end
    return depth == 0 ? maxd : -1
end

"Find one '()' and turn it into ')('; returns nothing if none found."
function swap_one_pair(s::AbstractString)
    for i in 1:(lastindex(s)-1)
        if s[i] == '(' && s[i+1] == ')'
            return s[1:i-1] * ")(" * (i+1 < lastindex(s) ? s[i+2:end] : "")
        end
    end
    return nothing
end

@inline function random_char_index(rng::AbstractRNG, s::AbstractString)
    steps = rand(rng, 1:length(s))  # length(s) counts characters
    i = firstindex(s)
    for _ in 2:steps
        i = nextind(s, i)
    end
    return i
end

"STRICT corruption: flip exactly one character (guarantees unbalanced, length preserved)."
function corrupt_strict(rng::AbstractRNG, s::String)::String
    @assert !isempty(s)
    i = random_char_index(rng, s)
    c = s[i]
    @assert (c == '(' || c == ')')
    newc = (c == '(') ? ')' : '('
    li, ri = firstindex(s), lastindex(s)
    left  = (i > li) ? s[li:prevind(s, i)] : ""
    right = (i < ri) ? s[nextind(s, i):ri] : ""
    return string(left, newc, right)
end

"Dataset with zero label-noise for negatives (uses `corrupt_strict`)."
function make_dataset_strict(rng::AbstractRNG; N::Int=10_000, n_range::UnitRange{Int}=2:8, negative_ratio::Float64=0.5)
    xs = Vector{String}(undef, N)
    ys = Vector{Int}(undef, N)
    for i in 1:N
        is_positive = rand(rng) > negative_ratio
        n = rand(rng, n_range)
        s = balanced_string(rng, n)
        if is_positive
            xs[i] = s; ys[i] = 1
        else
            xs[i] = corrupt_strict(rng, s); ys[i] = 0
        end
    end
    return xs, ys
end

# ---------------------------
# Data loader utilities
# ---------------------------

"Map '(' -> 1, ')' -> 2."
@inline function tokenize_parens(s::AbstractString)::Vector{Int}
    out = Vector{Int}(undef, length(s))
    @inbounds for i in 1:length(s)
        out[i] = (s[i] == '(') ? 1 : 2
    end
    return out
end

"Tokenize a vector of strings."
@inline batch_tokenize(xs::Vector{String}) = [tokenize_parens(x) for x in xs]

"Pad variable-length token sequences to a (max_len, batch) Int matrix with `pad_id` (default 0)."
function pad_sequences(seqs::Vector{Vector{Int}}; pad_id::Int=0)
    @assert !isempty(seqs) "pad_sequences received an empty sequence list."
    max_len = maximum(length, seqs)
    batch   = length(seqs)
    X = fill(pad_id, max_len, batch)
    @inbounds for (j, v) in enumerate(seqs)
        X[1:length(v), j] = v
    end
    return X
end

"Convenience: create a boolean mask (same shape as X) where true = non-pad."
@inline make_mask(X::AbstractMatrix{<:Integer}; pad_id::Int=0) = X .!= pad_id

"""
Create minibatches from raw string/label arrays.

Returns a `Vector{Tuple{Matrix{Int}, Vector{Int}}}` where each tuple is `(X, y)`.
- `X` has shape (max_len_in_batch, batch_size)
- `y` is a Vector{Int} of labels in {0,1}
"""
function make_minibatches(xs::Vector{String}, ys::Vector{Int};
                          batch_size::Int=64, shuffle::Bool=true,
                          rng::AbstractRNG=Random.GLOBAL_RNG, pad_id::Int=0)
    @assert length(xs) == length(ys) "xs and ys must be same length."
    n = length(xs)
    indices = collect(1:n)
    if shuffle
        Random.shuffle!(rng, indices)
    end
    batches = Vector{Tuple{Matrix{Int}, Vector{Int}}}()
    for k in 1:batch_size:n
        sel = indices[k:min(k+batch_size-1, n)]
        toks = batch_tokenize(xs[sel])
        X = pad_sequences(toks; pad_id=pad_id)
        y = ys[sel]
        push!(batches, (X, y))
    end
    return batches
end

"""
Produce ID/MID/OOD splits.

- strict=false  → uses make_dataset (may have tiny noise in negatives)
- strict=true, strict_mode=:flip       → uses make_dataset_strict (flip one char)
- strict=true, strict_mode=:equalcount → uses make_dataset_strict_equalcount (harder)
"""
function make_splits(rng::AbstractRNG;
                     Ntrain::Int=20_000, Nval::Int=2_000, Nmid::Int=2_000, Nood::Int=2_000,
                     id_range::UnitRange{Int}=2:8, mid_range::UnitRange{Int}=9:10, ood_range::UnitRange{Int}=12:14,
                     negative_ratio::Float64=0.5, strict::Bool=true, strict_mode::Symbol=:equalcount)
    maker = if !strict
        make_dataset
    elseif strict_mode === :equalcount
        make_dataset_strict_equalcount
    elseif strict_mode === :flip
        make_dataset_strict
    else
        error("unknown strict_mode=$strict_mode (use :equalcount or :flip)")
    end
    train_x, train_y = maker(rng; N=Ntrain, n_range=id_range,  negative_ratio=negative_ratio)
    val_x,   val_y   = maker(rng; N=Nval,   n_range=id_range,  negative_ratio=negative_ratio)
    mid_x,   mid_y   = maker(rng; N=Nmid,   n_range=mid_range, negative_ratio=negative_ratio)
    ood_x,   ood_y   = maker(rng; N=Nood,   n_range=ood_range, negative_ratio=negative_ratio)
    return (; train_x, train_y, val_x, val_y, mid_x, mid_y, ood_x, ood_y)
end



function corrupt_strict_equalcount(s::String)::String
    @assert !isempty(s)
    @assert s[1] == '(' && s[end] == ')'  # true for any nonempty Dyck-1 string
    return ")" * (length(s) > 2 ? s[2:end-1] : "") * "("
end

"Dataset with zero label-noise for negatives AND equal '(' / ')' counts."
function make_dataset_strict_equalcount(rng::AbstractRNG; N::Int=10_000,
                                        n_range::UnitRange{Int}=2:8,
                                        negative_ratio::Float64=0.5)
    xs = Vector{String}(undef, N)
    ys = Vector{Int}(undef, N)
    for i in 1:N
        is_positive = rand(rng) > negative_ratio
        n = rand(rng, n_range)
        s = balanced_string(rng, n)
        if is_positive
            xs[i] = s; ys[i] = 1
        else
            xs[i] = corrupt_strict_equalcount(s); ys[i] = 0
        end
    end
    return xs, ys
end

end # module
