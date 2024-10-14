abstract type Node end

mutable struct BranchNode <: Node
    split_var::UInt16
    cutpoint::Float64
    left::Node
    right::Node

end

mutable struct Leaf <: Node
    μ::Float64
    #Σr::Float64 --- This might not be necessary the Leaf should store only the \mu value.
    #∑r²::Float64 --- Same
end

mutable struct Tree
    root::Node
end

function get_leaf_nodes(node::Node)

     if(typeof(node)==Leaf)
        [node]
     else 
        reduce(vcat,[get_leaf_nodes(node.left),get_leaf_nodes(node.right)])
     end
end

# This function initialise the search of the node by the root for the tree
#(remember that the leaves do not store any information about the parents so, we save memory in this way)
function get_my_parent(node::Node, tree::Tree)
    get_my_parent(node,tree.root)
end

# It is important to notice that this will check if the evaluated parent (starting from the root)
# as a candidate of the node that we want to evaluate;
function get_my_parent(node::Node, parent_candidate::Branch)
    if (node==parent_candidate.left) || (node==parent_candidate.right)
        return parent_candidate
    else 
        isa(get_my_parent(node,parent_candidate.left),Nothing)?
            get_my_parent(node,parent_candidate.right):get_my_parent(node,parent_candidate.left)
    end
end 

# Simple, if the parent candidate is a Leaf, it cannot be a parent!
function get_my_parent(node::Node, parent_candidate::Leaf)
    return nothing
end


