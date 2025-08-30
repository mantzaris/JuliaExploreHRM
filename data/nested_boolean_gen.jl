
module BooleanDataGenerator

using Random

abstract type BoolNode end
struct VarNode <: BoolNode; var_index::Int; end
struct NotNode <: BoolNode; child::BoolNode; end
struct BinNode <: BoolNode; operation::Symbol; left::BoolNode; right::BoolNode; end

function generate_boolean_tree(rng::AbstractRNG; min_depth::Int=2, max_depth::Int=4, 
                              variable_count::Int=5, held_out_ops::Set{Symbol}=Set{Symbol}())
    all_ops = (:AND, :OR, :XOR, :NAND)
    available_ops = [op for op in all_ops if op ∉ held_out_ops]
    isempty(available_ops) && error("No operations available")
    
    function gen(depth::Int)
        if depth <= 1
            return VarNode(rand(rng, 1:variable_count))
        end
        
        choice = rand(rng, [:var, :not, :op, :op, :op])  # Bias toward ops
        
        if choice == :var
            return VarNode(rand(rng, 1:variable_count))
        elseif choice == :not
            return NotNode(gen(depth - 1))
        else
            op = rand(rng, available_ops)
            left_depth = rand(rng, 1:(depth-1))
            right_depth = rand(rng, 1:(depth-1))
            return BinNode(op, gen(left_depth), gen(right_depth))
        end
    end
    
    for _ in 1:100
        target_depth = rand(rng, min_depth:max_depth)
        tree = gen(target_depth)
        if min_depth <= tree_depth(tree) <= max_depth
            return tree
        end
    end
    
    # Fallback
    return BinNode(first(available_ops), VarNode(1), VarNode(2))
end

function tree_depth(node::BoolNode)::Int
    if node isa VarNode
        return 1
    elseif node isa NotNode
        return 1 + tree_depth(node.child)
    else
        return 1 + max(tree_depth(node.left), tree_depth(node.right))
    end
end

function eval_boolean_tree(node::BoolNode, var_values::Vector{Bool})::Bool
    if node isa VarNode
        return var_values[node.var_index]
    elseif node isa NotNode
        return !eval_boolean_tree(node.child, var_values)
    else
        left_val = eval_boolean_tree(node.left, var_values)
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
            error("Unknown operation")
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

"""
Generate boolean formula dataset.

Parameters:
- n: Number of samples
- variable_count: Number of variables (default 5)
- min_depth, max_depth: Tree depth range (default 2-4)
- held_out_ops: Operations to exclude (default none)
- seed: Random seed (default 42)

Returns:
- X: Matrix of variable assignments (n x variable_count)
- y: Vector of labels (n,)
- expressions: Vector of expression strings (n,)
"""
function generate_data(n::Int=100; 
                      variable_count::Int=5,
                      min_depth::Int=2, 
                      max_depth::Int=4,
                      held_out_ops::Vector{Symbol}=Symbol[],
                      seed::Int=42)
    
    rng = MersenneTwister(seed)
    held_out_set = Set(held_out_ops)
    
    X = zeros(Int, n, variable_count)
    y = zeros(Int, n)
    expressions = String[]
    
    for i in 1:n
        tree = generate_boolean_tree(rng; 
            min_depth=min_depth, 
            max_depth=max_depth,
            variable_count=variable_count,
            held_out_ops=held_out_set)
        
        var_vals = rand(rng, Bool, variable_count)
        result = eval_boolean_tree(tree, var_vals)
        expr = boolean_tree_to_prefix(tree)
        
        X[i, :] = [v ? 1 : 0 for v in var_vals]
        y[i] = result ? 1 : 0
        push!(expressions, expr)
    end
    
    return X, y, expressions
end

export generate_data

end

