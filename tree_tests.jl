x = reshape([0.1; 0.5; 0.9],3,1)
valid_cutpoint_ = true
root = Branch(1,0.5,Branch(1,td_.xcut[2,1],Leaf(0.0),Leaf(0.0)),Leaf(0.0))
tree = Tree(root)

matrix_test = leafprob(x,tree)