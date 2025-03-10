using LinearAlgebra
using Statistics
include("tree.jl")
include("bart_model.jl")
include("calculating_tree_probabilities.jl")
include("utils.jl")
import Random
Random.seed!(42)

# Sample data
n_ = 10000
x  = rand(n_,2)
y = 5 .+ 2 .*x[:,1] .+ randn(n_)              # Response vector#
y = reshape(y, :, 1)        # Explicitly reshape y into a 4x1 matrix

td_::TrainData = TrainData(x,y,100,false)
hypers_::Hypers =  Hypers(td_)
mcmc_::MCMC = MCMC()


bm_::BartModel = BartModel(hypers_,td_,mcmc_)

