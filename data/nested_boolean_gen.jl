
module BooleanDataGenerator

using Random

abstract type BoolNode end
struct VarNode <: BoolNode; var_index::Int; end
struct NotNode <: BoolNode; child::BoolNode; end
struct BinNode <: BoolNode; operation::Symbol; left::BoolNode; right::BoolNode; end


function generate_boolean_tree(rng::AbstractRNG; min_depth::Int=2, max_depth::Int=4,
                               variable_count::Int=5, held_out_ops::Set{Symbol}=Set{Symbol}(),
                               mode::Symbol=:exact)
    @assert min_depth >= 1 "min_depth must be >= 1"
    @assert max_depth >= min_depth "max_depth must be >= min_depth"
    @assert variable_count >= 1 "variable_count must be >= 1"

    all_ops = (:AND, :OR, :XOR, :NAND)
    available_ops = [op for op in all_ops if op ∉ held_out_ops]
    isempty(available_ops) && error("No operations available")

    # --- helper to draw a variable index safely
    draw_var() = VarNode(rand(rng, 1:variable_count))

    # --- EXACT mode: guarantee depth == target_depth by construction
    function gen_exact(depth::Int)::BoolNode
        if depth == 1
            return draw_var()
        end
        # Disallow early :var. Choose between unary NOT or binary op.
        choice = rand(rng, [:not, :op, :op])  # bias to binary
        if choice == :not
            # NOT must propagate the "must reach" budget
            return NotNode(gen_exact(depth - 1))
        else
            op = rand(rng, available_ops)
            # Ensure one branch *must* reach depth-1; the other can be anything (loose)
            if rand(rng, Bool)
                left  = gen_exact(depth - 1)
                # right is allowed to be smaller
                right = gen_loose(rand(rng, 1:(depth - 1)))
                return BinNode(op, left, right)
            else
                left  = gen_loose(rand(rng, 1:(depth - 1)))
                right = gen_exact(depth - 1)
                return BinNode(op, left, right)
            end
        end
    end

    # --- LOOSE mode: your original stochastic program (slightly refactored)
    function gen_loose(depth::Int)::BoolNode
        if depth <= 1
            return draw_var()
        end
        choice = rand(rng, [:var, :not, :op, :op, :op])  # bias toward ops
        if choice == :var
            return draw_var()
        elseif choice == :not
            return NotNode(gen_loose(depth - 1))
        else
            op = rand(rng, available_ops)
            left_depth  = rand(rng, 1:(depth - 1))
            right_depth = rand(rng, 1:(depth - 1))
            return BinNode(op, gen_loose(left_depth), gen_loose(right_depth))
        end
    end

    # --- main logic
    target_depth = rand(rng, min_depth:max_depth)
    if mode === :exact
        return gen_exact(target_depth)
    elseif mode === :loose
        # rejection sampling as in your code, but with safe fallback
        for _ in 1:200  # allow a few more tries for deep ranges
            tree = gen_loose(target_depth)
            d = tree_depth(tree)
            if min_depth <= d <= max_depth
                return tree
            end
        end
        # Safe fallback that never violates variable_count
        if variable_count == 1
            return NotNode(VarNode(1))
        else
            return BinNode(first(available_ops), VarNode(1), VarNode(2))
        end
    else
        error("Unknown mode = $mode; use :loose or :exact")
    end
end



function tree_depth(node::BoolNode)::Int
    if node isa VarNode
        return 1
    elseif node isa NotNode
        return 1 + tree_depth(node.child)
    elseif node isa BinNode
        return 1 + max(tree_depth(node.left), tree_depth(node.right))
    else
        error("Unknown node type")
    end
end

function eval_boolean_tree(node::BoolNode, var_values::Vector{Bool})::Bool
    if node isa VarNode
        return var_values[node.var_index]
    elseif node isa NotNode
        return !eval_boolean_tree(node.child, var_values)
    else
        left_val  = eval_boolean_tree(node.left, var_values)
        right_val = eval_boolean_tree(node.right, var_values)
        if node.operation === :AND
            return left_val & right_val
        elseif node.operation === :OR
            return left_val | right_val
        elseif node.operation === :XOR
            return xor(left_val, right_val)
        elseif node.operation === :NAND
            return !(left_val & right_val)
        else
            error("Unknown operation $(node.operation)")
        end
    end
end

function boolean_tree_to_prefix(node::BoolNode)::String
    if node isa VarNode
        return "x$(node.var_index)"
    elseif node isa NotNode
        return "(NOT $(boolean_tree_to_prefix(node.child)))"
    else
        return "($(node.operation) $(boolean_tree_to_prefix(node.left)) $(boolean_tree_to_prefix(node.right)))"
    end
end


function generate_data(n::Int=100;
                       variable_count::Int=5,
                       min_depth::Int=2,
                       max_depth::Int=4,
                       held_out_ops::Vector{Symbol}=Symbol[],
                       seed::Int=42,
                       mode::Symbol=:loose)

    rng = MersenneTwister(seed)
    held_out_set = Set(held_out_ops)

    X = zeros(Int, n, variable_count)
    y = zeros(Int, n)
    expressions = String[]

    for i in 1:n
        tree = generate_boolean_tree(rng;
            min_depth=min_depth, max_depth=max_depth,
            variable_count=variable_count, held_out_ops=held_out_set,
            mode=mode)

        var_vals = rand(rng, Bool, variable_count)
        result   = eval_boolean_tree(tree, var_vals)
        expr     = boolean_tree_to_prefix(tree)

        X[i, :] = [v ? 1 : 0 for v in var_vals]
        y[i]    = result ? 1 : 0
        push!(expressions, expr)
    end

    return X, y, expressions
end

export generate_data, generate_boolean_tree, tree_depth,
       eval_boolean_tree, boolean_tree_to_prefix


end

