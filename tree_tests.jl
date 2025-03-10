x = reshape([1.1; 2.1; 3.1],3,1)
valid_cutpoint_ = true
root = Branch(1,0.5,Branch(1,td_.xcut[1,1],Leaf(0.0),Leaf(0.0)),Leaf(0.0))
tree = Tree(root)

leafprob(x,tree)