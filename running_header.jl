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
Random.seed!(42)

# Sample data
n_ = 5
x  = rand(n_,2)
y = 5 .+ 2 .*x[:,1] .+ randn(n_)              # Response vector#
y = reshape(y, :, 1)        # Explicitly reshape y into a 4x1 matrix
tree_resid = rand(n_)
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


std_bart_state_ = StandardBartState(bart_ensemble_,tree_resid,1.0,[0.5,0.5])

# std_bart_state_ = StandardBartState(undef)
# ss_bart::BartSufficientStats = suffstats(b)
print(bart_tree.ss.number_leaves)
grow_proposal!(bart_tree,tree_resid,std_bart_state_,bm_)
print(bart_tree.ss.number_leaves)
# prune_proposal!(bart_tree,tree_resid,std_bart_state_,bm_)
draw_μ!(bart_tree,std_bart_state_)
print(bart_tree.ss.number_leaves)
# prune_proposal!(bart_tree,tree_resid,std_bart_state_,bm_)