using Distributions

mutable struct SufficientStats
    number_leaves::Int
    S::Matrix{Float64}

end

    

struct BartTree

    tree::Tree
    X_tree::Matrix{Float64} # This is the matrix of indicators to represent the terminal node structure. It has dimensions n × number_leaves
    ss::SufficientStats 

end

struct TrainData
    n::Int64
    p::Int64
    x_train::Matrix{Float64}
    y_train::AbstractMatrix
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
    nchain::Int64
    nthin::Int64
    npost::Int64

    function MCMC(;niter = 2500, nburn = 500, nchain = 1, nthin = 1)
        new(niter,nburn,nchain,nthin,niter-nburn)
    end

end


struct Hypers
    m::Int64
    α::Float64
    β::Float64
    k::Float64
    σ_μ::Float64
    q::Float64
    ν::Float64
    λ::Float64

    function Hypers(td::TrainData; m = 200,α = 0.95, β = 2.0,k = 2.0, q = 0.9,ν = 3)

        # Quicker and clever way of obtaing
        λ = 1/quantile(InverseGamma(ν/2,ν/(2*td.σ_OLS^2)),q)

        if isa(td.y_train,Matrix{Int})
            σ_μ = sqrt(9/(m*k^2))
        else 
            σ_μ = sqrt((maximum(td.y_train)-minimum(td.y_train))/(4*m*k^2))
        end
                
        new(m,α,β,k,σ_μ,q,ν,λ)
    end              
end


struct BartModel
    hypers::Hypers
    td::TrainData
    mcmc::MCMC
end



function TrainData(x_train::Matrix{Float64},y_train::AbstractMatrix,numcut::Int64,usequant::Bool)
    n = length(y_train)
    p = size(x_train,2)
    xmin = minimum(x_train,dims = 1)
    xmax = maximum(x_train,dims = 1)
    ymin = minimum(y_train)
    ymax = maximum(y_train)
    scale_X!(x_train,xmin,xmax)
    if isa(y_train,Matrix{Float64})
        normalize_y!(y_train,ymin,ymax)
    end

    σ_OLS = naive_sigma(x_train,y_train)
    if usequant
        xcut = get_xcut(x_train,numcut)
    else 
        xcut = get_xcut(x_train,xmin,xmax,numcut)
    end

    TrainData(n,p,x_train,y_train,xmin,xmax,xcut,ymin,ymax,σ_OLS)
end

##### sampler types
abstract type BartState end