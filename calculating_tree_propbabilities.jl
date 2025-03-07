# Probability of a node being terminal
function get_prob_non_terminal(depth::Int64, bm::BartModel)
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


# Getting the probability of going to left (here we have hard-boundaries yet so it will be assigned one)
function probleft(x::Vector{Float64},branch::Branch)
    x[branch.split_var]<= branch.cutpoint ? 1.0 : 0.0
end

function probleft(X::Matrix{Float64},branch::Branch)
    ifelse.(X[:,branch.split_var]<= branch.cutpoint ? 1.0 : 0.0)
end


function leafprob(x::Vector{Float64},tree::Tree)
    if isa(tree.root,Leaf)
        return 1.0
    end 
    X_tree = Float64[]
    goesleft = probleft(x, tree.root)
    goesright = 1.0 - goesleft
    leafprob(x, tree.root.left,tree,goesleft,X_tree)
    leafprob(x, tree.root.right,tree,goesright,X_tree)
end

function leafprob(x::Vector{Float64}, branch::Branch, tree::Tree,current_prob::Float64, X_tree::Vector{Float64})
    goesleft = current_prob * probleft(x,branch)
    goesright = current_prob - goesleft
    leafprob(x,branch.left,tree,goesleft,X_tree)
    leafprob(x,branch.right,tree,goesright,X_tree)
end

function leafprob(x::Vector{Float64}, leaf::Leaf, tree::Tree, current_prob::Float64, X_tree::Vector{Float64})
    push!(X_tree,current_prob)
end

# This function initialise the creating of the X_tree, remember each row is a observation and each column is a terminal node
function leafprob(X::Matrix{Float64},bt::BartTree,bm::BartModel)::Matrix{Float64}
    X_tree::Matrix{Float64} = zeros(Float64,bm.td.n,bt.ss.number_leaves)
    for i in 1:bm.td.n
        X_tree[i,:] .= leafprob(X[i,:],bt.tree)
    end
    return X_tree
end

function leafprob(X::Matrix{Float64},tree::Tree)::Matrix{Float64}
    n::Int64 = size(X,1)
    X_tree::Matrix{Float64} = zeros(Float64,n,length(get_leaf_nodes(tree.root)))
    for i in 1:n
        X_tree[i,:] .= leafprob(X[i,:],tree)
    end
    return X_tree
end

# Functions associated with the Dirichlet prior from linero

# If the arrives into a Leaf it will do nothing ( do not count in leaves)
varcount!(leaf::Leaf,counts::Vector) = nothing

# Iterating over the tree counting all the rules
function varcount!(branch::Branch,counts::Vector)
    var_counts[branch.split_var] += 1
    varcount!(branch.left,var_counts)
    varcount!(branch.right,var_counts)
end

# Count within a single tree
function varcount(tree::Tree, bm::BartModel)
    var_counts = zeros(bm.td.p)
    var_counts = varcount!(tree.root,var_counts)
    return var_counts
end

## Counting for all trees:
function varcounts(trees::Vector{BartTree}, bm::BartModel)
    vec(sum(reduce(hcat, [varcount(bt.tree, bm) for bt in trees]), dims = 2))
end

# Function helper to avoid numerical issues
function log_sum_exp(x)
    m::Float64 = maximum(x)
    return m + log(sum(exp.(x .- m)))
  end

  
# Updating the vector of probabilities for each covariate
function draws!(bs::BartState, bm::BartModel)
    
    # Count variables for the whole forest
    forest_vc = varcounts(bs.ensemble.trees,bm::BartModel)
    shapes = bs.shape / bm.td.p .+ forest_vc # Recall that bs.shape is the dirichelt parameter \alpha in Linero original paper
    y = log.(rand.(Gamma.(shapes .+ 1, 1))) # Applying the log to avoid numerical issues
    z = log.(rand(length(shapes))) ./ shapes # Adding a uniform noise trick
    logs = y + z
    logs = logs .- log_sum_exp(logs) # This subtraction does the normalization
    bs.s = logs # Logs of the probability of sampling a preditor j from p -- need to remember to exponetiate when samping a var
end    

## Sampling a var from the updated bs.s from a dirichelt from linero paper
function sample_var(bs::BartState,bm::BartModel)
    return sample(1:bm.td.p, weights(exp.(bs.s)))
end