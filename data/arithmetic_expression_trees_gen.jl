module ArithmeticExpressionData
using Random

abstract type ExprNode end
struct ConstNode <: ExprNode
    value::Int
end
struct BinExprNode <: ExprNode
    operator::Symbol
    left::ExprNode
    right::ExprNode
end

"Evaluate an expression tree safely; throw if result magnitude explodes."
function evaluate(node::ExprNode)::Int
    if node isa ConstNode
        return (node::ConstNode).value
    elseif node isa BinExprNode
        n = node::BinExprNode
        a = evaluate(n.left)
        b = evaluate(n.right)
        op = n.operator
        if op === :+
            return a + b
        elseif op === :-
            return a - b
        elseif op === :*
            return a * b
        else
            error("Unsupported operator: $op")
        end
    else
        error("Unknown node type")
    end
end

"Render to a fully parenthesized infix string."
function render_infix(node::ExprNode)::String
    if node isa ConstNode
        return string((node::ConstNode).value)
    else
        n = node::BinExprNode
        return "(" * render_infix(n.left) * string(n.operator) * render_infix(n.right) * ")"
    end
end

"Random expression with depth in [min_depth, max_depth]."
function generate_expression_tree(rng::AbstractRNG;
    min_depth::Int=2, max_depth::Int=4,
    operators::Vector{Symbol} = [:+, :-, :*],
    operand_range::UnitRange{Int}=0:9,
    max_abs_value::Int=10_000)::ExprNode

    @assert min_depth ≥ 1 && max_depth ≥ min_depth
    function gen(depth::Int)::ExprNode
        if depth == 1
            return ConstNode(rand(rng, operand_range))
        else
            # ensure we use internal nodes often enough
            op = rand(rng, operators)
            # randomly split remaining depth
            left_depth  = 1 + rand(rng, 0:depth-2)
            right_depth = depth - 1 - (left_depth - 1)
            left  = gen(left_depth)
            right = gen(right_depth)
            node = BinExprNode(op, left, right)
            # reject if value explodes
            try
                val = evaluate(node)
                if abs(val) > max_abs_value
                    return gen(depth) # resample
                end
            catch
                return gen(depth) # resample on error
            end
            return node
        end
    end
    return gen(rand(rng, min_depth:max_depth))
end

"Make a dataset of N examples. Returns (inputs, targets)."
function make_dataset(rng::AbstractRNG; N::Int=10_000,
    min_depth::Int=2, max_depth::Int=4,
    operators::Vector{Symbol}=[:+, :-, :*],
    operand_range::UnitRange{Int}=0:9)

    xs = Vector{String}(undef, N)
    ys = Vector{Int}(undef, N)
    for i in 1:N
        tree = generate_expression_tree(rng; min_depth=min_depth, max_depth=max_depth,
                                        operators=operators, operand_range=operand_range)
        xs[i] = render_infix(tree)
        ys[i] = evaluate(tree)
    end
    return xs, ys
end

end # module
