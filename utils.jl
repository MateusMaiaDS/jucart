function scale_X!(x_train::Matrix{Float64},xmin::Matrix{Float64},xmax::Matrix{Float64})
    x_train .= (x_train.-xmin)./(xmax .- xmin)
end