using LinearAlgebra

include("bart_model.jl")
include("tree.jl")
include("calculating_tree_propbabilities.jl")
include("utils.jl")


# Sample data
x = [1.0 2.0; 3.0 4.0; 3.2 6.0; 7.0 8.0]  # Predictor matrix (4 observations, 2 predictors)
y = [1.5, 3.5, 5.5, 7.5]                   # Response vector#
y = reshape(y, :, 1)        # Explicitly reshape y into a 4x1 matrix
