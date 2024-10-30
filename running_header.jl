using LinearAlgebra

include("bart_model.jl")
include("tree.jl")
include("calculating_tree_propbabilities.jl")
include("utils.jl")


# Sample data
n_ = 10000
x  = rand(n_,2)
y = 5 .+ 2 .*x[:,1] .+ randn(n_)              # Response vector#
y = reshape(y, :, 1)        # Explicitly reshape y into a 4x1 matrix

td_::TrainData = TrainData(x,y,100,false)
hypers_::Hypers =  Hypers(td_)

