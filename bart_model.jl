using Distributions


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
    τ_OLS::Float64
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
    τ_μ::Float64
    q::Float64
    a_τ::Float64
    d_τ::Float64
    function Hypers(td::TrainData; m = 200,α = 0.95, β = 2.0,k = 2.0, q = 0.9,a_τ = 3)
        qchi = quantile(Chisq(a_τ),1-q)
        lambda = (td.τ_OLS^(-1)*qchi)/a_τ
        d_τ = lambda*a_τ/2

        if isa(td.y_train,Matrix{Int})
            τ_μ = (m*k^2)/9
        else 
            τ_μ = (4*m*k^2)/(maximum(td.y_train)-minimum(td.y_train))
        end
        
        new(m,α,β,k,τ_μ,q,a_τ,d_τ)
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

    τ_OLS = naive_tau(x_train,y_train)
    if usequant
        xcut = get_xcut(x_train,numcut)
    else 
        xcut = get_xcut(x_train,xmin,xmax,numcut)
    end

    TrainData(n,p,x_train,y_train,xmin,xmax,xcut,ymin,ymax,τ_OLS)
end

