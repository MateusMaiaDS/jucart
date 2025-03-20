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
function grow_proposal!(bart_tree::BartTree, tree_residuals::Vector{Float64},bart_state::BartState,bart_model::BartModel)

    leaves::Vector{Node} = get_leaf_nodes(bart_tree.tree.root)
    index = rand(1:length(leaves))
    leaf = leaves[index]

    new_variable = sample_var(bart_state,bart_model)
    new_cutpoint = draw_cutpoint!(leaf,new_variable,bart_tree.tree,bart_model)
    # Updating the sufficent statistics of the current tree
    bart_tree.ss = get_suffstats(tree_residuals,bart_tree.X_tree,bart_state,bart_model)

    # Ending funcion for invalid grow
    if new_cutpoint==-1
        return 
    end

    branch = Branch(new_variable,new_cutpoint,Leaf(0.0),Leaf(0.0))

    goesleft = bart_tree.X_tree[:,index] .* probleft(bart_model.td.x_train,branch)
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
        X_tree_prime = zeros(bart_model.td.n,length(leaves)+1)
        X_tree_prime[:,indexes] = hcat(goesleft,goesright)
        X_tree_prime[:,setdiff(1:end,indexes)] = bart_tree.X_tree[:,setdiff(1:end,index)] # This is the easiest way to add new columns from the old one, as the index can be at any position not only on the end
    end

    ss_prime = get_suffstats(tree_residuals,X_tree_prime,bart_state,bart_model)
    mloglikratio = mll(ss_prime,bart_state,bart_model,indexes) - mll(bart_tree.ss,bart_state,bart_model,index) # Here is the key difference, because we are in BART, actually there's only a change on the index and indexes columns otherwise we would need to iterate over all columns
    # mloglikratio = mll(tree_residuals,X_tree_prime,bart_state,bart_model) - mll(tree_residuals,tree.X_tree,bart_state,bart_model) # SoftBART version
    treeratio = get_log_prob_grow_leaf_ratio(leaf,bart_tree.tree,bart_model)
    
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
        else 
            parent_node.right = Leaf(0.0)
        end 
    end 

end


# Growing a node
function prune_proposal!(bart_tree::BartTree, tree_residuals::Vector{Float64},bart_state::BartState,bart_model::BartModel)


    # Updating the sufficent statistics of the current tree
    # (need to update in any case as they are used to sample \mu)
    bart_tree.ss = get_suffstats(tree_residuals,bart_tree.X_tree,bart_state,bart_model)
    
    # Getting only parents of terminal nodes
    branch = rand(get_onlyparents(bart_tree.tree))

    indexes =  sort(findall(x -> (x == branch.left) || (x == branch.right), get_leaf_nodes(bart_tree.tree.root))) # Before I did in a way that I would collect only leaves, and selecting it's parent. But can be the case it's sibling isn't a terminal node.
    
    X_tree_prime = copy(bart_tree.X_tree)
    X_tree_prime[:,indexes[1]] = sum(bart_tree.X_tree[:,indexes],dims = 2 )
    X_tree_prime = X_tree_prime[:,1:end .!= indexes[2]]

    
    ss_prime = get_suffstats(tree_residuals,X_tree_prime,bart_state,bart_model)
    mloglikratio = mll(ss_prime,bart_state,bart_model,indexes[1]) - mll(bart_tree.ss,bart_state,bart_model,indexes) # Here is the key difference, because we are in BART, actually there's only a change on the index and indexes columns otherwise we would need to iterate over all columns
    # mloglikratio = mll(tree_residuals,X_tree_prime,bart_state,bart_model) - mll(tree_residuals,tree.X_tree,bart_state,bart_model) # SoftBART version
    treeratio = get_log_prob_prune_branch_ratio(branch,bart_model,bart_tree.tree)
    transratio = get_log_prune_trans_ratio(bart_tree,X_tree_prime)
    logratio = mloglikratio + treeratio + transratio

    if log(rand()) < logratio
        death_branch!(branch,bart_tree.tree)
        bart_tree.X_tree = X_tree_prime
        bart_tree.ss = ss_prime
    end
end

function change_branch!(branch::Branch,tree::Tree,new_split_var::Int64,new_cutpoint::Float64)
    
    if tree.root == branch
        tree.root.split_var = new_split_var
        tree.root.cutpoint = new_cutpoint
    else 
        parent_node = get_my_parent(branch,tree)

        if parent_node.left == branch
            parent_node.left.split_var = new_split_var
            parent_node.left.cutpoint = new_cutpoint
        else parent_node.right == branch 
            parent_node.right.split_var = new_split_var
            parent_node.right.cutpoint = new_cutpoint
        end 
    end 

end

# Adding the change verb
function change_proposal!(bart_tree::BartTree,tree_residuals::Vector{Float64},bart_state::BartState,bart_model::BartModel)

    bart_tree.ss = get_suffstats(tree_residuals,bart_tree.X_tree,bart_state,bart_model)

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
    new_variable = sample_var(bart_state,bart_model)
    new_cutpoint = draw_cutpoint!(branch,new_variable,bart_tree.tree,bart_model)
    
    new_branch = deepcopy(branch)
    new_branch.split_var = new_variable
    new_branch.cutpoint = new_cutpoint

    X_tree_parent = sum(bart_tree.X_tree[:,indexes],dims = 2)[:,1]
    goesleft = X_tree_parent .* probleft(bart_model.td.x_train,new_branch)
    goesright = X_tree_parent .- goesleft

    X_tree_prime = copy(bart_tree.X_tree)
    X_tree_prime[:,indexes] = hcat(goesleft,goesright)

    ss_prime = get_suffstats(tree_residuals,X_tree_prime,bart_state,bart_model)

    mloglikratio = mll(ss_prime,bart_state,bart_model,indexes) - mll(bart_tree.ss,bart_state,bart_model,indexes) # Here is the key difference, because we are in BART, actually there's only a change on the index and indexes columns otherwise we would need to iterate over all columns
   
    logratio = mloglikratio 

    if log(rand()) < logratio
        # Unnecessary navigation over the tree (old-version): # change_branch!(branch,bart_tree.tree,new_variable,new_cutpoint)
        branch.split_var = new_variable
        branch.cutpoint = new_cutpoint
        bart_tree.X_tree = X_tree_prime
        bart_tree.ss = ss_prime
    end

end

