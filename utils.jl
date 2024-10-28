function scale_X!(x_train::Matrix{Float64},xmin::Matrix{Float64},xmax::Matrix{Float64})
    x_train .= (x_train.-xmin)./(xmax .- xmin)
end

function normalize_y!(y_train::Matrix{Float64}, ymin::Float64,ymax::Float64)
    y_train .= (y_train .- ymin)./(ymax.-ymin) .- 0.5
end

function unnormalize_y!(y_train::Matrix{Float64}, ymin::Float64,ymax::Float64)
    y_train .= (ymax.-ymin).*(y_train .+ 0.5)./ .- ymin
end

function naive_tau(x_train::Matrix{Float64},y_train::Matrix{Float64})

    n_  = size(x_train,1)
    p_ = size(x_train,2)

    X_intercept::Matrix{Float64} = hcat(ones(n_),x_train)

    β = (X_intercept'*X_intercept)\(X_intercept'*y_train)
    y_hat = X_intercept*β
    residuals = y_train - y_hat
    
    RSS = dot(residuals,residuals)
    sigma_squared = RSS/(n_-p_-1)

    return 1/sigma_squared
end

