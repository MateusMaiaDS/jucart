# Probability of a node being terminal
function get_prob_non_terminal(depth::Int64, bart_model::BartModel)
    bart_model.hypers.α*(1+depth)^(-bart_model.hypers.β)
end

# Probability ratio from a MH step from a GROW move ( no node to calculate the whole tree)
function get_log_prob_grow_leaf_ratio(leaf::Node,tree::Tree, bart_model::BartModel)

    depth = get_depth(leaf,tree)
    prob_selected_leaf_nonterminal::Float64 = get_prob_non_terminal(depth,bart_model) # Calculate here before to avoid calculate twice later
    return (log(prob_selected_leaf_nonterminal)+ 2*log(1-get_prob_non_terminal((depth+1),bart_model))) - log(1-prob_selected_leaf_nonterminal)

end

# Probability ratio from a MH step from a PRUNE move ( no node to calculate the whole tree)
function get_log_prob_prune_branch_ratio(branch::Branch, bart_model::BartModel,tree::Tree)

    depth = get_depth(branch,tree)

    prob_selected_branch_nonterminal::Float64 = get_prob_non_terminal(depth,bart_model) # Calculate here before to avoid calculate twice later

    return log(1-prob_selected_branch_nonterminal) - (log(prob_selected_branch_nonterminal) + 2*log(1-get_prob_non_terminal(depth+1,bart_model))) 
end


## Transition probabilities
function sample_grow_prob(tree::Tree)
    isa(tree.root,Leaf) ? 1.0 : (1/3) # Probability of Grow is 1/3 
end

function sample_grow_prob(X_tree::Matrix{Float64})
    size(X_tree,2)==1 ? 1.0 : (1/3) # Probability of Grow is 1/3 
end

function sample_prune_prob(tree::Tree)
    isa(tree.root,Leaf) ? 0.0 : (1/3) # Probability of Prune is 1/3 
end

function sample_prune_prob(X_tree::Matrix{Float64})
    size(X_tree,2)==1 ? 0.0 : (1/3) # Probability of Prune is 1/3 
end

function sample_change_prob(tree::Tree)
    isa(tree.root,Leaf) ? 0.0 : (1/3) # Probability of Prune is 1/3 
end

function sample_change_prob(X_tree::Matrix{Float64})
    size(X_tree,2)==1 ? 0.0 : (1/3) # Probability of Prune is 1/3 
end



function get_log_prune_trans_ratio(bart_tree::BartTree, X_tree_prime::Matrix{Float64})
    
    # Probability of transitioning from the Proposed tree to the original (i.e: from the Grown tree to the pruned)
    numr = (isa(bart_tree.tree.root,Leaf) ? 1.0 : (1/3)) / (bart_tree.ss.number_leaves-1) # Remember that the prune is not applied on the root-only tree so this denominator cannot be zero
    denomr = sample_prune_prob(X_tree_prime) / (length(get_onlyparents(bart_tree.tree)))

    log(numr) - log(denomr)
end

## The log ratio of the transition probabilities for a birth proposal
function get_log_grow_trans_ratio(bart_tree::BartTree, X_tree_prime::Matrix{Float64})
    # Probability of transitioning from proposed Tree back to the current Tree
    numr = sample_prune_prob(X_tree_prime) / (length(get_onlyparents(bart_tree.tree))) # The it would be length(search_onlyparents(bart_tree.tree)+1) only if Grow a node that create a symmetrical tree
    # Probability of transitioning from the current Tree to the proposed Tree
    denomr = sample_grow_prob(bart_tree.tree) / bart_tree.ss.number_leaves
    log(numr) - log(denomr)
end
  

# Getting the probability of going to left (here we have hard-boundaries yet so it will be assigned one)
function probleft(x::Vector{Float64},branch::Branch)
    x[branch.split_var] <= branch.cutpoint ? 1.0 : 0.0
end

function probleft(X::Matrix{Float64},branch::Branch)
    Float64.(X[:,branch.split_var] .<= branch.cutpoint )
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
function leafprob(X::Matrix{Float64},bt::BartTree,bart_model::BartModel)::Matrix{Float64}
    X_tree::Matrix{Float64} = zeros(Float64,bart_model.td.n,bt.ss.number_leaves)
    for i in 1:bart_model.td.n
        X_tree[i,:] .= leafprob(X[i,:],bt.tree)
    end
    return X_tree
end

function leafprob(X::Matrix{Float64},tree::Tree)::Matrix{Float64}
    n::Int64 = size(X,1)
    X_tree = zeros(Float64,n,length(get_leaf_nodes(tree.root)))
    for i in 1:n
        X_tree[i,:] .= leafprob(X[i,:],tree)
    end
    return X_tree
end

# Functions associated with the Dirichlet prior from linero
## == start == ##
# If the arrives into a Leaf it will do nothing ( do not count in leaves)
varcount!(leaf::Leaf,var_counts::Vector) = nothing

# Iterating over the tree counting all the rules
function varcount!(branch::Branch,var_counts::Vector)
    var_counts[branch.split_var] += 1
    varcount!(branch.left,var_counts)
    varcount!(branch.right,var_counts)
end

# Count within a single tree
function varcount(tree::Tree, bart_model::BartModel)
    var_counts = zeros(bart_model.td.p)
    varcount!(tree.root,var_counts)
    var_counts
end

## Counting for all trees:
function varcounts(trees::Vector{BartTree}, bart_model::BartModel)
    comprehension_list = [varcount(bart_tree.tree, bart_model) for bart_tree in trees]
    vec(sum(reduce(hcat, comprehension_list), dims = 2))
end

# Function helper to avoid numerical issues
function log_sum_exp(x)
    m::Float64 = maximum(x)
    return m + log(sum(exp.(x .- m)))
  end

  
# Updating the vector of probabilities for each covariate
function draw_s!(bart_state::BartState, bart_model::BartModel)
    
    # Count variables for the whole forest
    forest_vc = varcounts(bart_state.ensemble.bart_trees,bart_model)
    shapes = bart_state.shapes_dirichlet / bart_model.td.p .+ forest_vc # Recall that bart_state.shape is the dirichelt parameter \alpha in Linero original paper
    y = log.(rand.(Gamma.(shapes .+ 1, 1))) # Applying the log to avoid numerical issues
    z = log.(rand(length(shapes))) ./ shapes # Adding a uniform noise trick
    logs = y + z
    logs = logs .- log_sum_exp(logs) # This subtraction does the normalization
    bart_state.s = logs # Logs of the probability of sampling a preditor j from p -- need to remember to exponetiate when samping a var
end    

## Sampling a var from the updated bart_state.s from a dirichelt from linero paper
function sample_var(bart_state::BartState,bart_model::BartModel)
    # Maybe need to add a conditional for the case when the Dirichlet prior isn't used 
    return sample(1:bart_model.td.p, weights(exp.(bart_state.s)))
end

## end -- linero functions

## Drawing a cutpoint for the proposed split_var
function draw_cutpoint!(leaf::Leaf,split_var::Int,tree::Tree,bart_model::BartModel)
    branch = leaf
    lower = [bart_model.td.xmin[:,split_var][1]]
    upper = [bart_model.td.xmax[:,split_var][1]]
    check = branch == tree.root ? false : true # Checking if arrived the root
    while check
        left = isLeft(branch,tree)
        branch = get_my_parent(branch,tree)
        check = branch == tree.root ? false : true
        if branch.split_var == split_var
            if left
                upper = push!(upper,branch.cutpoint)
            else 
                lower = push!(lower, branch.cutpoint)
            end
        end
    end
    lower = maximum(lower)
    upper = minimum(upper)

    # Using the xcut approach ( this is experimental using xcut, the alternative and simpler version is just a uniform sample -- commented below)
    mask = (bart_model.td.xcut[:,split_var] .> lower) .& (bart_model.td.xcut[:,split_var] .< upper)

    # Verifying if it's a valid cutpoint
    if(!any(mask))
        return -1.0 # Since data is always scaled there's no possible way of returning -1 (and this corresponds to an invalid node)
    else 
        return rand(bart_model.td.xcut[mask,split_var],1)[1]
    end
    
    # ## simpler version:
    # return (rand(Uniform(lower,upper)))
end

## Drawing a cutpoint for the proposed split_var
function draw_cutpoint!(branch::Branch,split_var::Int,tree::Tree,bart_model::BartModel)
    
    lower = [bart_model.td.xmin[:,split_var][1]]
    upper = [bart_model.td.xmax[:,split_var][1]]
    check = branch == tree.root ? false : true # Checking if arrived the root
    while check
        left = isLeft(branch,tree)
        branch = get_my_parent(branch,tree)
        check = branch == tree.root ? false : true
        if branch.split_var == split_var
            if left
                upper = push!(upper,branch.cutpoint)
            else 
                lower = push!(lower, branch.cutpoint)
            end
        end
    end
    lower = maximum(lower)
    upper = minimum(upper)

    # Using the xcut approach ( this is experimental using xcut, the alternative and simpler version is just a uniform sample -- commented below)
    mask = (bart_model.td.xcut[:,split_var] .> lower) .& (bart_model.td.xcut[:,split_var] .< upper)

    # # Verifying if it's a valid cutpoint
    if(!any(mask))
        return -1.0 # Since data is always scaled there's no possible way of returning -1 (and this corresponds to an invalid node)
    else 
        return rand(bart_model.td.xcut[mask,split_var],1)[1]
    end
    
    # ## simpler version:
    # return (rand(Uniform(lower,upper)))
end