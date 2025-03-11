# Growing a node
function grow_proposal!(tree::BartTree, tree_residuals::Vector{Float64},bs::BartState,bm::BartModel)

    leaves::Vector{Node} = get_leaf_nodes(tree.root)
    index = rand(1:length(leaves))
    leaf = leaves[index]

    new_variable = sample_var(bs,bm)
    new_cutpoint = drawcut(leaf,new_variable,tree,bm)
    # Updating the sufficent statistics of the current tree
    tree.ss = suffstats(tree_residuals,tree.X_tree,bs,bm)

    # Ending funcion for invalid grow
    if new_cutpoint==-1
        return 
    end

    branch = Branch(new_variable,new_cutpoint,Leaf(0.0),Leaf(0.0))
    goesleft = tree.X_tree[:,index] .* probleft(bm.td.x_train,branch,tree.tree)
    goesright = tree.X_tree[:,index] .- goesleft

    # Ending function for invalid grow
    if all(iszero,goesleft) | all(iszero,goesright) 
        return
    end

    if size(tree.X_tree,2)==1
        X_tree_prime = hcat(goesleft,goesright)
    else
        indexes = [index, index + 1]
        X_tree_prime = zeros(bm.td.n,length(leaves)+1)
        X_tree_prime[:,indexes] = hcat(goesleft,goesright)
        X_tree_prime[:,setdiff(1:end,indexes)] = tree.X_tree[setdiff(1:end,index)] # This is the easiest way to add new columns from the old one, as the index can be at any position not only on the end
    end

    ss_prime = suffstats(tree_residuals,X_tree_prime,bs,bm)
    mloglikratio = mll(ss_prime,bs,bm,indexes) - mll(tree.ss,bs,bm,index) # Here is the key difference, because we are in BART, actually there's only a change on the index and indexes columns otherwise we would need to iterate over all columns
    # mloglikratio = mll(tree_residuals,sX_tree_prime[:,indexes],bs,bm) - mll(tree_residuals,tree.X_tree[:,index],bs,bm) # SoftBART version
    treeratio = log_grow_tree(leaf,tree.tree,bm)
    transratio = log_grow_ratio(bt,X_tree_prime[:,indexes])
    logratio = mloglikratio + treeratio + transratio

    if log(rand()) < logratio
        birthleaf!(leaf,tree.tree,branch)
        tree.S = X_tree_prime
        tree.ss = ss_prime
    end

end