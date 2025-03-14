# Creating two children
function birth_leaf!(leaf::Leaf,tree::Tree,branch::Branch)
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

    new_cutpoint = draw_cutpoint!(leaf,new_variable,tree,bm)
    # Updating the sufficent statistics of the current tree
    bart_tree.ss = get_suffstats(tree_residuals,bart_tree.X_tree,bs,bm)

    # Ending funcion for invalid grow
    if new_cutpoint==-1
        return 
    end

    branch = Branch(new_variable,new_cutpoint,Leaf(0.0),Leaf(0.0))

    goesleft = bart_tree.X_tree[:,index] .* probleft(bm.td.x_train,branch)
    goesright = bart_tree.X_tree[:,index] .- goesleft


    # Ending function for invalid grow
    if all(iszero,goesleft) | all(iszero,goesright) 
        return
    end

    if size(bart_tree.X_tree,2)==1
        X_tree_prime = hcat(goesleft,goesright)
        indexes = [1, 2]
    else
        indexes = [index, index + 1]
        X_tree_prime = zeros(bm.td.n,length(leaves)+1)
        X_tree_prime[:,indexes] = hcat(goesleft,goesright)
        X_tree_prime[:,setdiff(1:end,indexes)] = bart_tree.X_tree[:,setdiff(1:end,index)] # This is the easiest way to add new columns from the old one, as the index can be at any position not only on the end
    end

    ss_prime = get_suffstats(tree_residuals,X_tree_prime,bs,bm)
    mloglikratio = mll(ss_prime,bs,bm,indexes) - mll(bart_tree.ss,bs,bm,index) # Here is the key difference, because we are in BART, actually there's only a change on the index and indexes columns otherwise we would need to iterate over all columns
    # mloglikratio = mll(tree_residuals,X_tree_prime,bs,bm) - mll(tree_residuals,tree.X_tree,bs,bm) # SoftBART version
    treeratio = get_log_prob_grow_leaf_ratio(leaf,bart_tree.tree,bm)
    
    # This step to have a proper calculation of the transition rate as the numerator can only use information from number of nogs of the new tree
    # new_tree = deepcopy(bart_tree.tree)
    # birth_leaf!(get_leaf_nodes(new_tree.root)[index],new_tree,branch)
    # transratio = get_log_grow_trans_ratio(new_tree,bart_tree,X_tree_prime)
    ### But this is skipped for computational purposes and assume the approximation of being equal to nogs of the current tree (it could also be nogs+1)

    transratio = get_log_grow_trans_ratio(bart_tree,X_tree_prime) # No need to make a copy for the new tree, if using the older version

    logratio = mloglikratio + treeratio + transratio

    if log(rand()) < logratio
        # bart_tree.tree = new_tree # Old version, with proper transition
        birth_leaf!(leaf,bart_tree.tree,branch) # Old version, with wrong transition
        bart_tree.X_tree = X_tree_prime
        bart_tree.ss = ss_prime
    end
end


function death_branch!(branch::Branch,tree::Tree)
    
    if tree.root == branch
        tree.root = Leaf(0.0)
    else 
        parent_node = get_my_parent(branch,tree)

        if parent_node.left == branch
            parent_node.left = Leaf(0.0)
        else parent_node.right == branch 
            parent_node.right = Leaf(0.0)
        end 
    end 

end


# Growing a node
function prune_proposal!(bart_tree::BartTree, tree_residuals::Vector{Float64},bs::BartState,bm::BartModel)


    # Updating the sufficent statistics of the current tree
    # (need to update in any case as they are used to sample \mu)
    bart_tree.ss = get_suffstats(tree_residuals,bart_tree.X_tree,bs,bm)
    leaves::Vector{Node} = get_leaf_nodes(bart_tree.tree.root)
    index = rand(1:length(leaves))
    leaf = leaves[index]
    leaf_isleft = isLeft(leaf,bart_tree.tree)
    
    if leaf_isleft
        indexes = [index, index + 1 ]
    else 
        indexes = [index - 1, index]
    end

    branch = get_my_parent(leaf,bart_tree.tree)

    X_tree_prime = copy(bart_tree.X_tree)
    X_tree_prime[:,indexes[1]] = sum(X_tree_prime[:,indexes],dims = 2 )
    X_tree_prime = X_tree_prime[:,1:end .!= indexes[2]]

    
    ss_prime = get_suffstats(tree_residuals,X_tree_prime,bs,bm)
    mloglikratio = mll(ss_prime,bs,bm,indexes[1]) - mll(bart_tree.ss,bs,bm,indexes) # Here is the key difference, because we are in BART, actually there's only a change on the index and indexes columns otherwise we would need to iterate over all columns
    # mloglikratio = mll(tree_residuals,X_tree_prime,bs,bm) - mll(tree_residuals,tree.X_tree,bs,bm) # SoftBART version
    treeratio = get_log_prob_prune_branch_ratio(branch,bm,bart_tree.tree)
    transratio = get_log_prune_trans_ratio(bart_tree,X_tree_prime)
    logratio = mloglikratio + treeratio + transratio

    if log(rand()) < logratio
        death_branch!(branch,bart_tree.tree)
        bart_tree.X_tree = X_tree_prime
        bart_tree.ss = ss_prime
    end
end

# Updating the μ
function draw_μ!(bart_tree::BartTree,bs::BartState)
    
    leaves = get_leaf_nodes(bart_tree.tree.root)

    for i in 1:length(leaves)
        leaves[i].μ = rand(Normal(bart_tree.ss.omega[i]*bart_tree.ss.r_sum[i],sqrt(bs.σ*bart_tree.ss.omega[i])),1)[1]
    end

end

