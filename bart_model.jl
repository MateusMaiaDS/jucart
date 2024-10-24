struct hypers
    α::Float64
    β::Float64
    τ_μ::Float64
    a_τ::Float64
    d_τ::Float64
end

struct BartModel
    hypers::hypers
    td::TrainData
end

struct TrainData
    x_train::Matrix{Float64}
    y_train::Matrix{Float64} 
end
