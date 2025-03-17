using Distributions

##### sampler types
abstract type BartState end

mutable struct SoftSufficientStats
    number_leaves::Int
    Ω::Matrix{Float64} # Who's S?
    rhat::Vector{Float64}
end

mutable struct BartSufficientStats <: SufficientStats
    number_leaves::Int
    omega::Vector{Float64} # Who's S?
    r_sum::Vector{Float64}

end
    

mutable struct BartTree

    tree::Tree
    X_tree::Matrix{Float64} # This is the matrix of indicators to represent the terminal node structure. It has dimensions n × number_leaves
    ss::SufficientStats 

end

struct SoftBartTree

    tree::Tree
    S::Matrix{Float64} # This is the matrix of indicators to represent the terminal node structure. It has dimensions n × number_leaves
    ss::SufficientStats 

end

struct TrainData
    n::Int64
    p::Int64
    x_train::Matrix{Float64}
    y_train::AbstractVector
    xmin::Matrix{Float64}
    xmax::Matrix{Float64}
    xcut::Matrix{Float64}
    ymin::Float64
    ymax::Float64
    σ_OLS::Float64
end

struct MCMC
    niter::Int64
    nburn::Int64
    nchains::Int64
    nthin::Int64

    function MCMC(;niter = 2500, nburn = 500, nchains = 1, nthin = 1)
        new(niter,nburn,nchains,nthin)
    end

end


struct Hypers
    m::Int64 # Number of trees
    α::Float64
    β::Float64
    k::Float64
    σ_μ::Float64
    q::Float64
    ν::Float64
    δ::Float64

    function Hypers(td::TrainData; m = 200,α = 0.95, β = 2.0,k = 2.0, q = 0.9,ν = 3)

        # Quicker and clever way of obtaing
        δ = 1/quantile(InverseGamma(ν/2,ν/(2*td.σ_OLS^2)),q)

        if isa(td.y_train,Vector{Int64})
            σ_μ = sqrt(9/(m*k^2))
        else 
            σ_μ = sqrt((maximum(td.y_train)-minimum(td.y_train))/(4*m*k^2))
        end
                
        new(m,α,β,k,σ_μ,q,ν,δ)
    end              
end


struct BartModel
    hypers::Hypers
    td::TrainData
    mcmc::MCMC
end

# Need to buiild a constructor for BartModel
function BartModel(X_train::Matrix{Float64},y_train::AbstractVector,mcmc::MCMC,numcut::Int64,usequant::Bool;hyperargs...)
    td = TrainData(X_train,y_train,numcut,usequant)
    hypers = Hypers(td;hyperargs...)
    BartModel(hypers,td,mcmc)
end

mutable struct BartEnsemble
    bart_trees::Vector{BartTree}
end

mutable struct StandardBartState <: BartState
    ensemble::BartEnsemble
    fhat::Vector{Float64}
    σ::Float64 # Residual standard deviation
    s::Vector{Float64} # Vector of probability of sampling a predictor
end

function StandardBartState(bart_model::BartModel)

    ## Need to flexbilize for multiple chains
    
    # Initializing the StandardBartState
    bart_trees = [BartTree(Tree(Leaf(0.0)),ones(Float64,bart_model.td.n,1),BartSufficientStats(1,ones(Float64,1),zeros(Float64,1))) for _ in 1:bart_model.hypers.m]
    init_f_hat = zeros(Float64,bart_model.td.n)

    return StandardBartState(bart_trees,init_f_hat,bart_model.td.σ_OLS,fill(1/bart_model.td.p,bart_model.td.p))

end


function TrainData(x_train::Matrix{Float64},y_train::AbstractVector,numcut::Int64,usequant::Bool)
    n = length(y_train)
    p = size(x_train,2)
    xmin = minimum(x_train,dims = 1)
    xmax = maximum(x_train,dims = 1)
    ymin = minimum(y_train)
    ymax = maximum(y_train)
    scale_X!(x_train,xmin,xmax)
    if isa(y_train,Vector{Float64})
        normalize_y!(y_train,ymin,ymax)
    end

    σ_OLS = naive_sigma(x_train,y_train)

    # ===  Not necessary really
    #  This would be used 
    if usequant
        xcut = get_xcut(x_train,numcut)
    else 
        xcut = get_xcut(x_train,xmin,xmax,numcut)
    end

    TrainData(n,p,x_train,y_train,xmin,xmax,xcut,ymin,ymax,σ_OLS)
end

