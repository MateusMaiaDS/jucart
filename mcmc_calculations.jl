function suffstats(tree_residuals::Vector{Float64},X_tree::Matrix{Float64},bs::StandardBartState,bm::BartModel)

    number_leaves = size(X_tree,2)
    omega = (bs.σ^(2)) ./ (sum(X_tree, dims = 1).*bm.hypers.σ_μ^2 .+ bs.σ^2) # A vector with σ²_μ/(σ²+nₗσ²_μ) for each terminal node
    r_sum = transpose(X_tree)*tree_residuals # Doing this operation is the same adding up residuals within terminal nodes ∑ᵢrᵢ

    BartSufficientStats(number_leaves,omega,r_sum)
end

function mll(ss::BartSufficientStats,bs::StandardBartState,bm::BartModel,indexes::Int)
    return -0.5*log(2*pi*bm.hypers.σ_μ^2)+0.5*log(bs.σ/ss.omega[indexes])+0.5*(ss.r_sum[indexes]^2)/(bs.σ*ss.omega[indexes])
end


function mll(ss::BartSufficientStats,bs::StandardBartState,bm::BartModel,indexes::Vector{Int})

    # Maybe throw an error here in the future when length(indexes)!=2
    left = (-0.5*log(2*pi*bm.hypers.σ_μ^2)+0.5*log(bs.σ/ss.omega[indexes[1]])+0.5*(ss.r_sum[indexes[1]]^2)/(bs.σ*ss.omega[indexes[1]])) 
    right = (-0.5*log(2*pi*bm.hypers.σ_μ^2)+0.5*log(bs.σ/ss.omega[indexes[2]])+0.5*(ss.r_sum[indexes[2]]^2)/(bs.σ*ss.omega[indexes[2]])) 
    return left + right

end


