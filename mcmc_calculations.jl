function get_suffstats(tree_residuals::Vector{Float64},X_tree::Matrix{Float64},bart_state::StandardBartState,bm::BartModel)

    number_leaves = size(X_tree,2)
    omega = (bart_state.σ^(2)) ./ (sum(X_tree, dims = 1).*bm.hypers.σ_μ^2 .+ bart_state.σ^2) # A vector with σ²_μ/(σ²+nₗσ²_μ) for each terminal node
    r_sum = transpose(X_tree)*tree_residuals # Doing this operation is the same adding up residuals within terminal nodes ∑ᵢrᵢ

    BartSufficientStats(number_leaves,omega[1,:],r_sum)
end

function mll(ss::BartSufficientStats,bart_state::StandardBartState,bm::BartModel,indexes::Int)
    return -0.5*log(2*pi*bm.hypers.σ_μ^2)+0.5*log(bart_state.σ/ss.omega[indexes])+0.5*(ss.r_sum[indexes]^2)/(bart_state.σ*ss.omega[indexes])
end


function mll(ss::BartSufficientStats,bart_state::StandardBartState,bm::BartModel,indexes::Vector{Int})

    # Maybe throw an error here in the future when length(indexes)!=2
    left = (-0.5*log(2*pi*bm.hypers.σ_μ^2)+0.5*log(bart_state.σ/ss.omega[indexes[1]])+0.5*(ss.r_sum[indexes[1]]^2)/(bart_state.σ*ss.omega[indexes[1]])) 
    right = (-0.5*log(2*pi*bm.hypers.σ_μ^2)+0.5*log(bart_state.σ/ss.omega[indexes[2]])+0.5*(ss.r_sum[indexes[2]]^2)/(bart_state.σ*ss.omega[indexes[2]])) 
    return left + right

end


# Updating the μ
function draw_μ!(bart_tree::BartTree,bart_state::BartState)
    
    leaves = get_leaf_nodes(bart_tree.tree.root)

    for i in 1:length(leaves)
        leaves[i].μ = rand(Normal(bart_tree.ss.omega[i]*bart_tree.ss.r_sum[i],sqrt(bart_state.σ*bart_tree.ss.omega[i])),1)[1]
    end

end

# Updating σ
function drawσ!(bart_state::BartState,bart_model::BartModel)

    a = 0.5*(bart_model.td.n + bart_model.hypers.ν)
    d = 0.5*(bart_model.hypers.ν + bart_model.hypers.δ + sum((bart_model.td.y_train-bart_state.fhat).^2))
    bart_state.σ = sqrt(rand(InverseGamma(a,d)))

end

function draw_tree!(bart_tree::BartTree, tree_residuals::Vector{Float64}, bart_state::BartState,bart_model::BartModel)

    verb_probs = Float64[sample_grow_prob(bart_tree.tree), sample_prune_prob(bart_tree.tree), sample_change_prob(bart_tree.tree)]
    if(sum(verb_probs)!=1.0)
        throw(DomainError("Sum of Tree moves proposal must be equal to one."))
    end
    tree_proposal = sample(['g','p','c',],weights(verb_probs))

    if tree_proposal == 'g'
        grow_proposal!(bart_tree,tree_residuals,bart_state,bart_model)
    elseif tree_proposal == 'p' 
        prune_proposal!(bart_tree,tree_residuals,bart_state,bart_model)
    elseif tree_proposal == 'c'
        change_proposal!(bart_tree,tree_residuals,bart_state,bart_model)
    else 
        throw(DomainError("Tree proposal is invalid",tree_proposal))
    end

end
function draw_trees!(bart_state::BartState,bart_model::BartModel)

    for bart_tree in bart_state.ensemble.trees
        fhat_without_current_tree = bart_state.fhat .- predict(bart_tree)
        tree_residuals = bart_model.td.y_train - fhat_without_current_tree
        draw_tree!(bart_tree,tree_residuals,bart_state,bart_model)
        draw_μ!(bart_tree,bart_state)
        bs.fhat = bs.fhat + predict(bayes_tree)
    end
end