function get_μ(tree::Tree)
    Float64[leaf.μ for leaf in get_leaf_nodes(tree.root)]
end

# Predicting for a single tree
function StatsBase.predict(bart_tree::BartTree)
    bart_tree.X_tree*get_μ(bart_tree.tree)
end

# Predicting for a ensemble of trees
function StatsBase.predict(bart_state::BartState,bart_model::BartModel)

    fhat = zeros(Float64,bart_model.td.n)

    for bart_tree in bart_state.ensemble.bart_trees
        fhat += predict(bart_tree)
    end

    fhat
end

# Predicting from a set of trees and X_tree
function StatsBase.predict(trees::Vector{Tree},X_matrix::Matrix{Float64})

    f_hat = zeros(Float64,size(X_matrix,1))
    
    for tree in trees
        f_hat += leafprob(X_matrix,tree)*get_μ(tree)
    end

    return f_hat
end