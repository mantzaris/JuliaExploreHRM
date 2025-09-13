module AdditionCarryData

using Random

export generate_addition_sum_dataset,
       generate_addition_verification_dataset,
       save_tsv,
       build_vocabulary,
       tokenize_with_vocabulary,
       digitsymbol

# ----------------------------
# Basic digit helpers
# ----------------------------

"""
    digitsymbol(d::Int)

Render a non-negative digit `d` (0-based) to a string token.
0-9 -> "0".."9"; 10->"A", 11->"B", ..., up to base ≤ 36.
"""
function digitsymbol(d::Int)
    d < 0 && error("Digit must be non-negative.")
    if d <= 9
        return string(d)
    else
        return string(Char('A' + (d - 10)))
    end
end

# ----------------------------
# Random number generation
# ----------------------------

"""
    rand_digits_msd(n; base=10, allow_leading_zero=false, rng=Random.default_rng())

Return an `n`-digit number as a vector of digits in **MSD->LSD** order.
- If `allow_leading_zero=false`, the most significant digit is sampled in 1:(base-1).
- Supports bases ≥2 and ≤36.
"""
function rand_digits_msd(n::Int; base::Int=10, allow_leading_zero::Bool=false,
                         rng::AbstractRNG=Random.default_rng())
    base < 2 && throw(ArgumentError("base must be ≥ 2"))
    base > 36 && throw(ArgumentError("base must be ≤ 36 for alphanumeric tokens"))
    if n == 0
        return Int[]
    end
    v = Vector{Int}(undef, n)
    if n == 1
        v[1] = rand(rng, 0:base-1)
        if !allow_leading_zero && v[1] == 0
            v[1] = rand(rng, 1:base-1)
        end
    else
        v[1] = allow_leading_zero ? rand(rng, 0:base-1) : rand(rng, 1:base-1)
        for i in 2:n
            v[i] = rand(rng, 0:base-1)
        end
    end
    return v
end

# ----------------------------
# Addition + carry analytics
# ----------------------------

"""
    add_msd(a_msd, b_msd; base=10) -> (c_msd, carries_lsd)

Add two digit vectors `a_msd` and `b_msd` (both MSD->LSD order) in given `base`.
Returns:
- `c_msd`: sum digits (MSD->LSD)
- `carries_lsd`: Bool vector (LSD->MSD) with carry flags at each position added
"""
function add_msd(a::Vector{Int}, b::Vector{Int}; base::Int=10)
    la, lb = length(a), length(b)
    n = max(la, lb)
    carry = 0
    c_rev = Vector{Int}()
    carries = Vector{Bool}()
    for k in 0:n-1
        da = (la - k >= 1) ? a[la - k] : 0
        db = (lb - k >= 1) ? b[lb - k] : 0
        s = da + db + carry
        push!(c_rev, s % base)
        newcarry = s ÷ base
        push!(carries, newcarry > 0)
        carry = newcarry
    end
    if carry > 0
        push!(c_rev, carry)
    end
    c_msd = reverse(c_rev)
    return c_msd, carries
end

"""
    max_carry_chain(carries_lsd) -> Int

Return the longest run of `true` in `carries_lsd` (which is LSD->MSD order).
"""
function max_carry_chain(carries::Vector{Bool})
    maxrun = 0
    run = 0
    for flag in carries
        if flag
            run += 1
            maxrun = max(maxrun, run)
        else
            run = 0
        end
    end
    return maxrun
end

# ----------------------------
# Serialization (tokens/strings)
# ----------------------------

"""
    serialize_digits(d_msd; order=:lsd_first, delimiter=" ", base=10, pad_to=nothing)

Render digits (MSD->LSD) as a space-separated token string.
- `order=:lsd_first` (default) prints LSD->MSD (useful for local step-by-step addition).
- `order=:msd_first` prints MSD->LSD.
- `pad_to` pads with zeros to a fixed length (right-pad for `:lsd_first`, left-pad for `:msd_first`).
"""
function serialize_digits(d_msd::Vector{Int}; order::Symbol=:lsd_first,
                          delimiter::String=" ", base::Int=10,
                          pad_to::Union{Int,Nothing}=nothing)
    digits = (order == :lsd_first) ? reverse(d_msd) : d_msd
    if pad_to !== nothing && length(digits) < pad_to
        if order == :lsd_first
            digits = vcat(digits, fill(0, pad_to - length(digits)))
        else
            digits = vcat(fill(0, pad_to - length(digits)), digits)
        end
    end
    return join(map(digitsymbol, digits), delimiter)
end

"""
    build_sum_input_string(a_msd, b_msd; base=10, order=:lsd_first,
                           delimiter=" ", include_plus=true,
                           include_equals=true, pad_a=nothing, pad_b=nothing)

Build the input sequence for the **sum** task:
    "A_tokens + B_tokens ="
(with configurable order/padding).
"""
function build_sum_input_string(a_msd::Vector{Int}, b_msd::Vector{Int};
                                base::Int=10, order::Symbol=:lsd_first,
                                delimiter::String=" ", include_plus::Bool=true,
                                include_equals::Bool=true,
                                pad_a::Union{Int,Nothing}=nothing,
                                pad_b::Union{Int,Nothing}=nothing)
    sa = serialize_digits(a_msd; order=order, delimiter=delimiter, base=base, pad_to=pad_a)
    sb = serialize_digits(b_msd; order=order, delimiter=delimiter, base=base, pad_to=pad_b)
    buf = IOBuffer()
    print(buf, sa, " ")
    if include_plus
        print(buf, "+ ")
    end
    print(buf, sb)
    if include_equals
        print(buf, " =")
    end
    return String(take!(buf))
end

# ----------------------------
# Vocabulary + tokenization
# ----------------------------

"""
    build_vocabulary(base; include_plus=true, include_equals=true,
                     include_pad=false, include_eos=false) -> Dict{String,Int}

Build a simple token->id vocabulary suitable for these tasks.
IDs are assigned deterministically in the order:
digits 0...(base-1), then "+", then "=", then optional specials.
"""
function build_vocabulary(base::Int; include_plus::Bool=true, include_equals::Bool=true,
                          include_pad::Bool=false, include_eos::Bool=false)
    base < 2 && throw(ArgumentError("base must be ≥ 2"))
    base > 36 && throw(ArgumentError("base must be ≤ 36 for alphanumeric tokens"))
    tokens = String[]
    for d in 0:base-1
        push!(tokens, digitsymbol(d))
    end
    if include_plus
        push!(tokens, "+")
    end
    if include_equals
        push!(tokens, "=")
    end
    if include_pad
        push!(tokens, "<pad>")
    end
    if include_eos
        push!(tokens, "<eos>")
    end
    vocab = Dict{String,Int}()
    for (i, t) in enumerate(tokens)
        vocab[t] = i
    end
    return vocab
end

"""
    tokenize_with_vocabulary(s::String, vocab::Dict{String,Int}) -> Vector{Int}

Split `s` on whitespace and map each token into an integer id using `vocab`.
Throws an error if an unknown token is encountered.
"""
function tokenize_with_vocabulary(s::String, vocab::Dict{String,Int})
    toks = split(s)
    ids = Vector{Int}(undef, length(toks))
    for (i, t) in enumerate(toks)
        haskey(vocab, t) || error("Unknown token '$t' in input string.")
        ids[i] = vocab[t]
    end
    return ids
end

# ----------------------------
# Negative candidate (for verification)
# ----------------------------

"""
    make_incorrect_c(correct_c_msd; base=10, rng=..., strategy=:single_digit) -> Vector{Int}

Generate an incorrect candidate sum for the verification task.
Strategies:
- `:single_digit` : change exactly one digit.
- `:near`         : add/subtract 1 on a random digit (bounded in [0, base-1]).
- `:random`       : random number with length in {L-1, L, L+1} (L = correct length).
"""
function make_incorrect_c(correct_c::Vector{Int}; base::Int=10,
                          rng::AbstractRNG=Random.default_rng(),
                          strategy::Symbol=:single_digit)
    if strategy == :single_digit
        c = copy(correct_c)
        pos = rand(rng, 1:length(c))
        old = c[pos]
        new_digit = rand(rng, 0:base-1)
        while new_digit == old
            new_digit = rand(rng, 0:base-1)
        end
        c[pos] = new_digit
        return c
    elseif strategy == :near
        c = copy(correct_c)
        pos = rand(rng, 1:length(c))
        old = c[pos]
        c[pos] = (old == base - 1) ? (old - 1) : (old + 1)
        return c
    elseif strategy == :random
        minlen = max(1, length(correct_c) - 1)
        maxlen = length(correct_c) + 1
        len = rand(rng, minlen:maxlen)
        c = rand_digits_msd(len; base=base, allow_leading_zero=true, rng=rng)
        if c == correct_c
            return make_incorrect_c(correct_c; base=base, rng=rng, strategy=:single_digit)
        end
        return c
    else
        error("Unknown negative strategy: $strategy")
    end
end

# ----------------------------
# Dataset generators
# ----------------------------

_choose_length(spec, rng) = spec isa Int ? spec : rand(rng, spec)

"""
    generate_addition_sum_dataset(n;
        digits_a=2, digits_b=2, base=10,
        random_seed=0, lsd_first=true, allow_leading_zero=false,
        require_carry=nothing, min_carry_chain=0,
        pad_width_a=nothing, pad_width_b=nothing, pad_width_c=nothing,
        delimiter=" ", include_plus=true, include_equals=true,
        max_tries_per_example=1000)

Return a `Vector{NamedTuple}` where each item has:
- `input::String`  — e.g., "3 7 + 2 9 ="
- `target::String` — e.g., "5 1 1"
- `a,b,c::Vector{Int}` (MSD->LSD),
- `carry_positions::Vector{Bool}` (LSD->MSD),
- `max_carry_chain::Int`, `base::Int`

You can constrain examples to have (or not have) carries, and/or a minimum carry-chain length.
"""
function generate_addition_sum_dataset(num_samples::Int;
    digits_a::Union{Int,UnitRange{Int}}=2,
    digits_b::Union{Int,UnitRange{Int}}=2,
    base::Int=10,
    random_seed::Union{Nothing,Int}=0,
    lsd_first::Bool=true,
    allow_leading_zero::Bool=false,
    require_carry::Union{Bool,Nothing}=nothing,
    min_carry_chain::Int=0,
    pad_width_a::Union{Nothing,Int}=nothing,
    pad_width_b::Union{Nothing,Int}=nothing,
    pad_width_c::Union{Nothing,Int}=nothing,
    delimiter::String=" ",
    include_plus::Bool=true,
    include_equals::Bool=true,
    max_tries_per_example::Int=1000
)
    rng = isnothing(random_seed) ? Random.default_rng() : MersenneTwister(random_seed)
    order = lsd_first ? :lsd_first : :msd_first
    results = Vector{NamedTuple}(undef, 0)
    count = 0
    tries = 0
    while count < num_samples
        tries += 1
        if tries > max(num_samples * 10, max_tries_per_example * num_samples)
            error("Could not satisfy constraints; relax carry constraints or increase max_tries_per_example.")
        end
        la = _choose_length(digits_a, rng)
        lb = _choose_length(digits_b, rng)
        a = rand_digits_msd(la; base=base, allow_leading_zero=allow_leading_zero, rng=rng)
        b = rand_digits_msd(lb; base=base, allow_leading_zero=allow_leading_zero, rng=rng)
        c, carries = add_msd(a, b; base=base)
        mchain = max_carry_chain(carries)
        if !isnothing(require_carry)
            if require_carry && mchain == 0; continue; end
            if (!require_carry) && mchain > 0; continue; end
        end
        if mchain < min_carry_chain; continue; end
        input_str = build_sum_input_string(a, b; base=base, order=order, delimiter=delimiter,
                                           include_plus=include_plus, include_equals=include_equals,
                                           pad_a=pad_width_a, pad_b=pad_width_b)
        target_str = serialize_digits(c; base=base, order=order, delimiter=delimiter, pad_to=pad_width_c)
        push!(results, (input=input_str, target=target_str, a=a, b=b, c=c, base=base,
                        carry_positions=carries, max_carry_chain=mchain))
        count += 1
    end
    return results
end

"""
    generate_addition_verification_dataset(n;
        digits_a=2, digits_b=2, base=10, positive_fraction=0.5,
        negative_strategy=:single_digit,
        random_seed=0, lsd_first=true, allow_leading_zero=false,
        require_carry=nothing, min_carry_chain=0,
        pad_width_a=nothing, pad_width_b=nothing, pad_width_c=nothing,
        delimiter=" ", include_plus=true, include_equals=true,
        max_tries_per_example=1000)

Return a `Vector{NamedTuple}` where each item has:
- `input::String` — e.g., "3 7 + 2 9 = 5 1 1"
- `label::Bool`   — `true` if the equation is correct
- metadata: `a,b,c::Vector{Int}`, `carry_positions`, `max_carry_chain`, `base`
"""
function generate_addition_verification_dataset(num_samples::Int;
    digits_a::Union{Int,UnitRange{Int}}=2,
    digits_b::Union{Int,UnitRange{Int}}=2,
    base::Int=10,
    positive_fraction::Float64=0.5,
    negative_strategy::Symbol=:single_digit,
    random_seed::Union{Nothing,Int}=0,
    lsd_first::Bool=true,
    allow_leading_zero::Bool=false,
    require_carry::Union{Bool,Nothing}=nothing,
    min_carry_chain::Int=0,
    pad_width_a::Union{Nothing,Int}=nothing,
    pad_width_b::Union{Nothing,Int}=nothing,
    pad_width_c::Union{Nothing,Int}=nothing,
    delimiter::String=" ",
    include_plus::Bool=true,
    include_equals::Bool=true,
    max_tries_per_example::Int=1000
)
    rng = isnothing(random_seed) ? Random.default_rng() : MersenneTwister(random_seed)
    order = lsd_first ? :lsd_first : :msd_first
    results = Vector{NamedTuple}(undef, 0)
    target_positive = round(Int, positive_fraction * num_samples)
    n_pos = 0
    n_neg = 0
    tries = 0
    while n_pos + n_neg < num_samples
        tries += 1
        if tries > max(num_samples * 10, max_tries_per_example * num_samples)
            error("Could not satisfy constraints; relax carry constraints or increase max_tries_per_example.")
        end
        la = _choose_length(digits_a, rng)
        lb = _choose_length(digits_b, rng)
        a = rand_digits_msd(la; base=base, allow_leading_zero=allow_leading_zero, rng=rng)
        b = rand_digits_msd(lb; base=base, allow_leading_zero=allow_leading_zero, rng=rng)
        correct_c, carries = add_msd(a, b; base=base)
        mchain = max_carry_chain(carries)
        if !isnothing(require_carry)
            if require_carry && mchain == 0; continue; end
            if (!require_carry) && mchain > 0; continue; end
        end
        if mchain < min_carry_chain; continue; end

        is_positive = (n_pos < target_positive)
        c = is_positive ? correct_c :
            make_incorrect_c(correct_c; base=base, rng=rng, strategy=negative_strategy)
        if (!is_positive) && c == correct_c
            continue
        end

        sa = serialize_digits(a; order=order, delimiter=delimiter, base=base, pad_to=pad_width_a)
        sb = serialize_digits(b; order=order, delimiter=delimiter, base=base, pad_to=pad_width_b)
        sc = serialize_digits(c; order=order, delimiter=delimiter, base=base, pad_to=pad_width_c)

        buf = IOBuffer()
        print(buf, sa, " ")
        if include_plus
            print(buf, "+ ")
        end
        print(buf, sb)
        if include_equals
            print(buf, " = ")
        else
            print(buf, " ")
        end
        print(buf, sc)
        input_str = String(take!(buf))

        label = is_positive
        push!(results, (input=input_str, label=label, a=a, b=b, c=c, base=base,
                        carry_positions=carries, max_carry_chain=mchain))
        if is_positive
            n_pos += 1
        else
            n_neg += 1
        end
    end
    return results
end

# ----------------------------
# Persistence (TSV)
# ----------------------------

repr_for_tsv(v) = v isa Vector{Int}  ? join(v, ",") :
                  v isa Vector{Bool} ? join((b ? "1" : "0" for b in v), ",") :
                  v isa Bool         ? (v ? "true" : "false") :
                  string(v)

"""
    save_tsv(path, records)

Write a TSV with columns drawn from the union of fields across all records.
Priority order: input, target, label, a, b, c, base, max_carry_chain, carry_positions, ...
Digit vectors and carry vectors are serialized as comma-separated lists.
"""
function save_tsv(path::AbstractString, records::Vector{NamedTuple})
    isempty(records) && error("No records to save.")
    # union of fields, with a consistent priority order
    priority = [:input, :target, :label, :a, :b, :c, :base, :max_carry_chain, :carry_positions]
    present = Set{Symbol}()
    for r in records
        for k in keys(r)
            push!(present, k)
        end
    end
    header = Symbol[]
    for p in priority
        if p in present
            push!(header, p)
        end
    end
    # any extra fields in alphabetical order
    for s in sort(collect(setdiff(present, Set(priority))))
        push!(header, s)
    end

    open(path, "w") do io
        println(io, join(string.(header), '\t'))
        for r in records
            row = String[]
            for h in header
                push!(row, haskey(r, h) ? repr_for_tsv(r[h]) : "")
            end
            println(io, join(row, '\t'))
        end
    end
    return nothing
end









# --- Simple ID / MID / OOD builders for long-addition (sum) ---
# Drop this at the end of AdditionCarryData.jl, before `end # module`

export make_addition_sum_splits, write_addition_sum_splits

"""
    make_addition_sum_splits(; kwargs...) -> NamedTuple

Builds five datasets for seq2seq addition (sum target):
  - train_id, val_id, test_id  (same length regime)
  - test_mid                   (slightly longer)
  - test_ood                   (much longer)

Defaults: pure length shift; no carry constraints unless you set them.
"""
function make_addition_sum_splits(;
    # sizes
    n_train::Int=100_000,
    n_val::Int=10_000,
    n_test_id::Int=10_000,
    n_test_mid::Int=10_000,
    n_test_ood::Int=10_000,

    # length regimes (digits in A and B)
    train_digits_a::UnitRange{Int}=2:4,
    train_digits_b::UnitRange{Int}=2:4,
    mid_digits_a::UnitRange{Int}=5:5,
    mid_digits_b::UnitRange{Int}=5:5,
    ood_digits_a::UnitRange{Int}=6:7,
    ood_digits_b::UnitRange{Int}=6:7,

    # arithmetic / formatting
    base::Int=10,
    lsd_first::Bool=true,
    allow_leading_zero::Bool=false,
    delimiter::String=" ",
    include_plus::Bool=true,
    include_equals::Bool=true,

    # (optional) carry controls — keep at 0 for pure length shift
    require_carry::Union{Bool,Nothing}=nothing,
    min_carry_chain_id::Int=0,
    min_carry_chain_mid::Int=0,
    min_carry_chain_ood::Int=0,

    # (optional) fixed-width padding
    pad_width_a::Union{Int,Nothing}=nothing,
    pad_width_b::Union{Int,Nothing}=nothing,
    pad_width_c::Union{Int,Nothing}=nothing,

    # reproducibility
    seed::Int=1234
)
    ds_train = generate_addition_sum_dataset(n_train;
        digits_a=train_digits_a, digits_b=train_digits_b, base=base,
        random_seed=seed, lsd_first=lsd_first, allow_leading_zero=allow_leading_zero,
        require_carry=require_carry, min_carry_chain=min_carry_chain_id,
        pad_width_a=pad_width_a, pad_width_b=pad_width_b, pad_width_c=pad_width_c,
        delimiter=delimiter, include_plus=include_plus, include_equals=include_equals)

    ds_val = generate_addition_sum_dataset(n_val;
        digits_a=train_digits_a, digits_b=train_digits_b, base=base,
        random_seed=seed+1, lsd_first=lsd_first, allow_leading_zero=allow_leading_zero,
        require_carry=require_carry, min_carry_chain=min_carry_chain_id,
        pad_width_a=pad_width_a, pad_width_b=pad_width_b, pad_width_c=pad_width_c,
        delimiter=delimiter, include_plus=include_plus, include_equals=include_equals)

    ds_id = generate_addition_sum_dataset(n_test_id;
        digits_a=train_digits_a, digits_b=train_digits_b, base=base,
        random_seed=seed+2, lsd_first=lsd_first, allow_leading_zero=allow_leading_zero,
        require_carry=require_carry, min_carry_chain=min_carry_chain_id,
        pad_width_a=pad_width_a, pad_width_b=pad_width_b, pad_width_c=pad_width_c,
        delimiter=delimiter, include_plus=include_plus, include_equals=include_equals)

    ds_mid = generate_addition_sum_dataset(n_test_mid;
        digits_a=mid_digits_a, digits_b=mid_digits_b, base=base,
        random_seed=seed+3, lsd_first=lsd_first, allow_leading_zero=allow_leading_zero,
        require_carry=require_carry, min_carry_chain=min_carry_chain_mid,
        pad_width_a=pad_width_a, pad_width_b=pad_width_b, pad_width_c=pad_width_c,
        delimiter=delimiter, include_plus=include_plus, include_equals=include_equals)

    ds_ood = generate_addition_sum_dataset(n_test_ood;
        digits_a=ood_digits_a, digits_b=ood_digits_b, base=base,
        random_seed=seed+4, lsd_first=lsd_first, allow_leading_zero=allow_leading_zero,
        require_carry=require_carry, min_carry_chain=min_carry_chain_ood,
        pad_width_a=pad_width_a, pad_width_b=pad_width_b, pad_width_c=pad_width_c,
        delimiter=delimiter, include_plus=include_plus, include_equals=include_equals)

    return (train_id=ds_train, val_id=ds_val, test_id=ds_id, test_mid=ds_mid, test_ood=ds_ood)
end

"""
    write_addition_sum_splits(dir, splits; base=10, include_vocab=true)

Writes:
  train_id.tsv, val_id.tsv, test_id.tsv, test_mid.tsv, test_ood.tsv
Optionally also writes a simple vocab.tsv for digits + "+" + "=".
"""
function write_addition_sum_splits(dir::AbstractString, splits;
    base::Int=10, include_vocab::Bool=true)
    isdir(dir) || mkpath(dir)
    save_tsv(joinpath(dir, "train_id.tsv"), splits.train_id)
    save_tsv(joinpath(dir, "val_id.tsv"),   splits.val_id)
    save_tsv(joinpath(dir, "test_id.tsv"),  splits.test_id)
    save_tsv(joinpath(dir, "test_mid.tsv"), splits.test_mid)
    save_tsv(joinpath(dir, "test_ood.tsv"), splits.test_ood)
    if include_vocab
        vocab = build_vocabulary(base; include_plus=true, include_equals=true)
        open(joinpath(dir, "vocab.tsv"), "w") do io
            println(io, "token\tid")
            for (tok, id) in sort(collect(vocab); by=x->x[2])
                println(io, "$(tok)\t$(id)")
            end
        end
    end
    return nothing
end




end # module






#####################
#####################

