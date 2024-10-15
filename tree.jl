abstract type Node end

mutable struct Branch <: Node
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
        isa(get_my_parent(node,parent_candidate.left),Nothing) ? get_my_parent(node,parent_candidate.right) : get_my_parent(node,parent_candidate.left)
    end
end 

# Simple, if the parent candidate is a Leaf, it cannot be a parent!
function get_my_parent(node::Node, parent_candidate::Leaf)
    return nothing
end

# Initialise the search of onlyparents for a tree
function search_onlyparents(tree::Tree)
    branches = Branch[]
    if isa(tree.root,Leaf) 
        return [tree.root] # Check this latter for me it would need to call nothing as there is no parents to search
    else 
        search_onlyparents(tree.root,branches)
    end
end

# Extremely intuitive, keep search for only parents nodes if there is always a branch
function search_onlyparents(branch::Branch,branches::Vector{Branch})
    if isa(branch.left,Leaf) && isa(branch.right,Leaf)
        push!(branches,branch)
    else 
        search_onlyparents(branch.left,branches)
        search_onlyparents(branch.right,branches)
    end 
    return branches
end

function search_onlyparents(leaf::Leaf,branches::Vector{Branch}) 
    return nothing
end


