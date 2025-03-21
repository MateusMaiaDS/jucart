using LinearAlgebra
using Statistics
using StatsBase
using Random
using Plots
using BenchmarkTools

include("tree.jl")
include("bart_model.jl")
include("calculating_tree_probabilities.jl")
include("utils.jl")
include("tree_moves.jl")
include("mcmc_calculations.jl")
include("predict.jl")
include("bart_main_function.jl")
Random.seed!(42)

# Sample data
n_ = 2000
# x  = reshape(rand(Uniform(0,1),n_),n_,1)
# y = vec(ifelse.(x .< 0.5,-1.0,1.0))  .+ rand(Normal(0,0.01),n_)
x  = reshape(rand(Uniform(-pi,pi),n_),n_,1)
y = vec(sin.(x))  .+ rand(Normal(0,0.1),n_)

numcut = 100; usequant = true; mcmc = MCMC()

# this is initialised only once and is not udpated
bart_model = BartModel(x,y,mcmc,numcut,usequant)

# UButuak ibe
bart_state = StandardBartState(bart_model)

# # Trying to keep growing a node
# count_i = 0
# # verb 
# bart_states = BartState[]
# post_fhat = zeros(Float64,1000,n_)
# for i in 1:1000
#     global count_i+=1
#     draw_trees!(bart_state,bart_model)
#     draw_σ!(bart_state,bart_model)
#     push!(bart_states,deepcopy(bart_state))
#     post_fhat[i,:] = bart_state.fhat
# end

# bart_run = fit_one(BartModel,x,y)

@btime fit_one(BartModel,x,y)

