# Probability of a node being terminal
function prob_non_terminal(depth::Int64, bm::BartModel)
    bm.hypers.α*(1+depth)^(-bm.hypers.β)
end

# Probability ratio from a MH step from a GROW move ( no node to calculate the whole tree)
function get_log_prob_grow_leaf(depth::Int64, bm::BartModel)

    prob_selected_leaf_nonterminal::Float64 = get_prob_non_terminal(depth,bm) # Calculate here before to avoid calculate twice later
    return (log(prob_selected_leaf_nonterminal)+ 2*log(1-get_prob_non_terminal((depth+1),bm))) - log(1-prob_selected_leaf_nonterminal)

end

# Probability ratio from a MH step from a PRUNE move ( no node to calculate the whole tree)
function get_log_prob_prune_node(depth::Int64, bm::BartModel)

    prob_selected_node_nonterminal::Float64 = get_prob_non_terminal(depth,bm) # Calculate here before to avoid calculate twice later

    return log(1-prob_selected_leaf_nonterminal) - (log(prob_selected_node_nonterminal) + 2*log(1-get_prob_non_terminal(depth+1,bm))) 
end

# Growing a node
function grow(tree::Tree,bm::BartModel)

    leaves::Vector{Node} = get_leaf_nodes(tree.root)
    index = rand(1:length(leaves))
    leaf = leaves[index]

    new_variable = sample_var()
    new_cutpoint = drawcut()

end

# Getting the probability of going to left (here we have hard-boundaries yet so it will be assigned one)
function probleft(x::Vector{Float64},branch::Branch)
    ifelse.(x[branch.split_var]<= branch.xcut ? 1.0 : 0.0)
end

function probleft(X::Vector{Float64},branch::Branch)
    ifelse.(X[:,branch.split_var]<= branch.xcut ? 1.0 : 0.0)
end


function leafprob(x::Vector{Float64},tree::Tree)
    if isa(tree.root,Leaf)
        1.0
    end 
    S = Float64[]
    goesleft = probleft(x, tree.root)
    goesright = 1.0 - goesleft
    leafprob(x, tree.root.left,tree,goesleft,S)
    leafprob(x, tree.root.right,tree,goesright,S)
end

function leafprob(x::Vector{Float64}, branch::Branch, tree::Tree,current_prob::Float64, S::Vector{Float64})
    goesleft = current_prob * probleft(x,branch)
    goesright = current_prob - goesleft
    leafprob(x,branch.left,tree,goesleft,S)
    leafprob(x,branch.right,tree,goesright,S)
end

function leafprob(x::Vector{Float64}, leaf::Leaf, tree::Tree, current_prob::Float64, S::Vector{Float64})
    push!(S,current_prob)
end

function leafprob(X::Matrix{Float64},bt::BartTree,bm::BartModel)
    S::Matrix{Float64} = zeros(Float64,bm.td.n,bt.ss.number_leaves)
    for i in 1:bm.td.n
        S[i,:] = leafprob(X[i,:],bt.tree)
    end
    S
end

function leafprob(X::Matrix{Float64},tree::Tree)::Matrix{Float64}
    n::Int64 = size(X,1)
    S::Matrix{Float64} = zeros(Float64,n,length(get_leaf_nodes(tree.root)))
    for i in 1:n
        S[i,:] = leafprob(X[i,:],tree)
    end
    S
end
