module Dyck1Data
using Random

"Generate a uniformly random balanced parentheses string with exact length `2n`."
function balanced_string(rng::AbstractRNG, n::Int)::String
    # Simple recursive Catalan sampler by random walk with validity check (not uniform over Catalan structures, but fine for ML datasets).
    s = IOBuffer()
    open_count = 0
    close_count = 0
    total = 2n
    for i in 1:total
        # we must ensure we never close more than open, and end with equal counts
        remaining = total - i + 1
        must_open  = open_count == close_count # cannot close first
        must_close = (open_count - close_count) == (remaining) # must close to finish
        if must_open
            print(s, '('); open_count += 1
        elseif must_close
            print(s, ')'); close_count += 1
        else
            if rand(rng) < 0.5
                print(s, '('); open_count += 1
            else
                # only close if we have unmatched opens
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

"Corrupt a balanced string by one of: drop a parenthesis, flip a parenthesis, or swap adjacent tokens."
function corrupt(rng::AbstractRNG, s::String)::String
    if length(s) == 0
        return ")"
    end
    mode = rand(rng, 1:3)
    if mode == 1
        # drop one char
        idx = rand(rng, 1:length(s))
        return s[1:idx-1] * (idx < length(s) ? s[idx+1:end] : "")
    elseif mode == 2
        # flip one char
        idx = rand(rng, 1:length(s))
        c = s[idx]
        flip = c == '(' ? ')' : '('
        return s[1:idx-1] * string(flip) * (idx < length(s) ? s[idx+1:end] : "")
    else
        # swap adjacent (if possible)
        if length(s) ≥ 2
            idx = rand(rng, 1:length(s)-1)
            return s[1:idx-1] * s[idx+1] * s[idx] * (idx+1 < length(s) ? s[idx+2:end] : "")
        else
            return s
        end
    end
end

"Make a labeled dataset: (strings, labels) where label=1 for balanced, 0 for unbalanced."
function make_dataset(rng::AbstractRNG; N::Int=10_000, n_range::UnitRange{Int}=2:8, negative_ratio::Float64=0.5)
    xs = Vector{String}(undef, N)
    ys = Vector{Int}(undef, N)
    for i in 1:N
        is_positive = rand(rng) > negative_ratio
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

end # module
