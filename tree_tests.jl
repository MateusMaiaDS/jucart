x = reshape([1.1; 2.1; 3.1],3,1)

root = Branch(1,2.4,Branch(1,1.0,Leaf(0.0),Leaf(0.0)),Leaf(0.0))
tree = Tree(root)

leafprob(x,tree)