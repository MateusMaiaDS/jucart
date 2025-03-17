function scale_X!(x_train::Matrix{Float64},xmin::Matrix{Float64},xmax::Matrix{Float64})
    x_train .= (x_train.-xmin)./(xmax .- xmin)
end

function normalize_y!(y_train::Matrix{Float64}, ymin::Vector{Float64},ymax::Vector{Float64})
    y_train .= (y_train .- ymin)./(ymax.-ymin) .- 0.5
end

function unnormalize_y!(y_train::Matrix{Float64}, ymin::Vector{Float64},ymax::Vector{Float64})
    y_train .= (ymax.-ymin).*(y_train .+ 0.5)./ .- ymin
end

function normalize_y!(y_train::Vector{Float64}, ymin::Float64,ymax::Float64)
    y_train .= (y_train .- ymin)./(ymax.-ymin) .- 0.5
end

function unnormalize_y!(y_train::Vector{Float64}, ymin::Float64,ymax::Float64)
    y_train .= (ymax.-ymin).*(y_train .+ 0.5)./ .- ymin
end

function naive_sigma(x_train::Matrix{Float64},y_train::Vector{Float64})

    n_  = size(x_train,1)
    p_ = size(x_train,2)

    X_intercept::Matrix{Float64} = hcat(ones(n_),x_train)

    β = (X_intercept'*X_intercept)\(X_intercept'*y_train)
    y_hat = X_intercept*β
    residuals = y_train - y_hat
    
    RSS = dot(residuals,residuals)
    sigma_naive = sqrt(RSS/(n_-p_-1))

    return sigma_naive
end

# ===========
# ( THIS ONE IS NOT REALLY USED IT )
# Getting the cutpoints matrix considering all variables scaled 
# ===========

function get_xcut(numcut::Int64)

    xcut::Matrix{Float64} = Matrix{Float64}(undef,numcut,1)

    xcut[:,1] = range(0,stop = 1, length = numcut+2)[2:(end-1)]

    return xcut

end

# Getting the cutpoints matrix considering different scales
function get_xcut(x_train::Matrix{Float64},xmin::Matrix{Float64},xmax::Matrix{Float64},numcut::Int64)

    xcut::Matrix{Float64} = Matrix{Float64}(undef,numcut,size(x_train,2))

    for j = 1:size(x_train,2)
        xcut[:,j] = range(xmin[1,j],stop = xmax[1,j], length = numcut+2)[2:(end-1)]
    end

    return xcut

end

# Getting a quantile version for it
function get_xcut(x_train::Matrix{Float64},numcut::Int64)

    xcut::Matrix{Float64} = Matrix{Float64}(undef,numcut,size(x_train,2))
    probs = collect(0:numcut+1) ./(numcut+1)
    for j = 1:size(x_train,2)
        xcut[:,j] = Statistics.quantile(x_train[:,j], probs)[2:(end-1)]
    end

    return xcut

end
