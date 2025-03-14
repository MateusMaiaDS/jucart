function get_μ(tree::Tree)
    Float64[leaf.μ for leaf in get_leaf_nodes(tree.root)]
end

# Predicting for a single tree
function StatsBase.predict(bart_tree::BartTree)
    return bart_tree.X_tree*get_μ(bart_tree.tree)
end

# Predicting for a ensemble of trees
function StatsBase.predict(bart_state::BartState,bm::BartModel)

    fhat = zeros(Float64,bm.td.n)

    for bart_tree in bart_state.ensemble.bart_trees
        fhat += predict(bart_tree)
    end

    return fhat
end