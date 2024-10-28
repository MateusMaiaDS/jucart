struct Hypers
    α::Float64
    β::Float64
    τ_μ::Float64
    a_τ::Float64
    d_τ::Float64
end

struct TrainData
    n::Int64
    p::Int64
    x_train::Matrix{Float64}
    y_train::Matrix{Float64} 
    xcut::Matrix{Float64}
    xmin::Matrix{Float64}
    xmax::Matrix{Float64}
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

struct BartModel
    hypers::Hypers
    td::TrainData
    mcmc::MCMC
end



function TrainData(x_train::Matrix{Float64},y_train::Matrix{Float64},numcut::Int64)
    n = length(y_train)
    p = size(x_train,2)
    xmin = minimum(x_train,dims = 1)
    xmax = maximum(x_train,dims = 1)
    ymin = minimum(y_train)
    ymax = maximum(y_train)
    #x_train = scale_X!(x_train,xmin,xmax)
    #y_train = normalize_y!(y_train,ymin,ymax)
    τ_OLS = naive_tau(x_train,y_train)
   TrainData(n,p,x_train,y_train,x_train,xmin,xmax,ymin,ymax,τ_OLS)
end

