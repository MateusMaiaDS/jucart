# Creating two children
function birthleaf!(leaf::Leaf,tree::Tree,branch::Branch)
    if isa(tree.root,Leaf)
        tree.root = branch
    else
        # Just a faster and concise way to look for the current leaf inside the current tree
        parent_node = get_my_parent(leaf,tree)
        
        # Notice that here by getting the parent node we are modifing the tree itself, 
        # and the branch object already contain the new split var and two-nnew leaves
        if parent_node.left==leaf
            parent_node.left = branch
        else
            parent_node.right = branch
        end
    end
end

# Growing a node
function grow_proposal!(bart_tree::BartTree, tree_residuals::Vector{Float64},bs::BartState,bm::BartModel)

    leaves::Vector{Node} = get_leaf_nodes(bart_tree.tree.root)
    index = rand(1:length(leaves))
    leaf = leaves[index]

    new_variable = sample_var(bs,bm)
    println("Success sample var")

    new_cutpoint = draw_cutpoint!(leaf,new_variable,tree,bm)
    println("Success newcutpoint")
    # Updating the sufficent statistics of the current tree
    bart_tree.ss = get_suffstats(tree_residuals,bart_tree.X_tree,bs,bm)
    println("Success suffstats")

    # Ending funcion for invalid grow
    if new_cutpoint==-1
        return 
    end

    branch = Branch(new_variable,new_cutpoint,Leaf(0.0),Leaf(0.0))

    println(branch)
    goesleft = bart_tree.X_tree[:,index] .* probleft(bm.td.x_train,branch)
    goesright = bart_tree.X_tree[:,index] .- goesleft

    print(goesleft)
    print(goesright)
    # Ending function for invalid grow
    if all(iszero,goesleft) | all(iszero,goesright) 
        return
    end

    if size(bart_tree.X_tree,2)==1
        X_tree_prime = hcat(goesleft,goesright)
    else
        indexes = [index, index + 1]
        X_tree_prime = zeros(bm.td.n,length(leaves)+1)
        X_tree_prime[:,indexes] = hcat(goesleft,goesright)
        X_tree_prime[:,setdiff(1:end,indexes)] = bart_tree.X_tree[:,setdiff(1:end,index)] # This is the easiest way to add new columns from the old one, as the index can be at any position not only on the end
    end

    ss_prime = get_suffstats(tree_residuals,X_tree_prime,bs,bm)
    mloglikratio = mll(ss_prime,bs,bm,indexes) - mll(bart_tree.ss,bs,bm,index) # Here is the key difference, because we are in BART, actually there's only a change on the index and indexes columns otherwise we would need to iterate over all columns
    # mloglikratio = mll(tree_residuals,sX_tree_prime[:,indexes],bs,bm) - mll(tree_residuals,tree.X_tree[:,index],bs,bm) # SoftBART version
    treeratio = get_log_prob_grow_leaf_ratio(leaf,bart_tree.tree,bm)
    transratio = get_log_grow_trans_ratio(bart_tree,X_tree_prime[:,indexes])
    logratio = mloglikratio + treeratio + transratio

    if log(rand()) < logratio
        birthleaf!(leaf,bart_tree.tree,branch)
        bart_tree.X_tree = X_tree_prime
        bart_tree.ss = ss_prime
    end

end