abstract type Node end

mutable struct BranchNode <: Node
    split_var::UInt16
    cutpoint::Float64
    left::Node
    right::Node

end

mutable struct Leaf <: Node
    μ::Float64
    Σr::Float64
    ∑r²::Float64
end

function get_leaf_nodes(node::Node)

     if(typeof(node)==Leaf)
        [node]
     else 
        reduce(vcat,[get_leaf_nodes(node.left),get_leaf_nodes(node.right)])
     end
end
