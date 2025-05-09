using LinearAlgebra
using Statistics
using StatsBase
using Random
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
n_ = 10
x  = reshape(collect(1:10) ./ 10,n_,1)
y = vec(ifelse.(x .< 0.5,-1.0,1.0))              # Response vector#
tree_resid = zeros(Float64,n_)
td_::TrainData = TrainData(x,y,100,false)
hypers_::Hypers =  Hypers(td_)
mcmc_::MCMC = MCMC()


bm_::BartModel = BartModel(hypers_,td_,mcmc_)

# This code initialises what would be an example of a BART fitted model 


root = Leaf(0.0)
tree = Tree(root)
matrix_test = reshape(ones(n_),:,1)
ss_bart = BartSufficientStats(1,[0.0],[0.0])
bart_tree = BartTree(tree,matrix_test,ss_bart)
bart_ensemble_ = BartEnsemble([bart_tree])


std_bart_state_ = StandardBartState(bart_ensemble_,tree_resid,1.0,[0.5,0.5],[1.0,1.0])

# std_bart_state_ = StandardBartState(undef)
# ss_bart::BartSufficientStats = suffstats(b)
# print(bart_tree.ss.number_leaves)
grow_proposal!(bart_tree,tree_resid,std_bart_state_,bm_)
# print(bart_tree.ss.number_leaves)
# prune_proposal!(bart_tree,tree_resid,std_bart_state_,bm_)
draw_μ!(bart_tree,std_bart_state_)
# print(bart_tree.ss.number_leaves)
# prune_proposal!(bart_tree,tree_resid,std_bart_state_,bm_)

bart_model_test = fit(BartModel,x,y)