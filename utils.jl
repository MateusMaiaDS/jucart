function scale_X!(x_train::Matrix{Float64},xmin::Matrix{Float64},xmax::Matrix{Float64})
    x_train .= (x_train.-xmin)./(xmax .- xmin)
end

function normalize_y!(y_train::Matrix{Float64}, ymin::Float64,ymax::Float64)
    y_train .= (y_train .- ymin)./(ymax.-ymin) .- 0.5
end

function unnormalize_y!(y_train::Matrix{Float64}, ymin::Float64,ymax::Float64)
    y_train .= (ymax.-ymin).*(y_train .+ 0.5)./ .- ymin
end

function naive_sigma!(traindata::TrainData)

    X_intercept::Matrix{Float64} = hcat(ones(traindata.n),traindata.x_train)

    β = (X_intercept'*X_intercept)\(X_intercept'*traindata.y)
    y_hat = X_intercept*β
    residuals = traindata.y_train - y_hat
    
    RSS = sum(residuals.^2)
    sigma_squared = RSS/(traindata.n-traindata.p-1)

    traindata.τ_OLS = 1/sigma_squared

    print(traindata.τ_OLS)

    return
end

