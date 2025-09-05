module LongAdditionData
using Random

"Generate two nonnegative integers with exactly `num_digits` digits each (no leading zeros)."
function random_ndigit(rng::AbstractRNG, num_digits::Int)::Int
    @assert num_digits ≥ 1
    lower = Int(10)^(num_digits-1)
    upper = Int(10)^num_digits - 1
    return rand(rng, lower:upper)
end

"Return (a_string, b_string, sum_string)."
function make_example(rng::AbstractRNG; num_digits::Int=3)
    a = random_ndigit(rng, num_digits)
    b = random_ndigit(rng, num_digits)
    s = a + b
    return string(a), string(b), string(s)
end

"Optionally also compute the per-column carry bits (right-to-left)."
function carries_for(a::String, b::String)
    maxlen = max(length(a), length(b))
    A = lpad(a, maxlen, '0')
    B = lpad(b, maxlen, '0')
    carry = 0
    carries = Vector{Int}(undef, maxlen)
    for i in maxlen:-1:1
        da = Int(A[i]) - Int('0')
        db = Int(B[i]) - Int('0')
        total = da + db + carry
        carries[i] = total ≥ 10 ? 1 : 0
        carry = carries[i]
    end
    return carries
end

"Make N examples. If `include_carries`, return a tuple of (a, b, sum, carries)."
function make_dataset(rng::AbstractRNG; N::Int=10_000, num_digits::Int=3, include_carries::Bool=false)
    a_strings = Vector{String}(undef, N)
    b_strings = Vector{String}(undef, N)
    sums      = Vector{String}(undef, N)
    carrybits = include_carries ? Vector{Vector{Int}}(undef, N) : Vector{Vector{Int}}()
    for i in 1:N
        a, b, s = make_example(rng; num_digits=num_digits)
        a_strings[i], b_strings[i], sums[i] = a, b, s
        if include_carries
            carrybits[i] = carries_for(a, b)
        end
    end
    return include_carries ? (a_strings, b_strings, sums, carrybits) : (a_strings, b_strings, sums)
end

end # module
