# Probability of a node being terminal
function prob_non_terminal(depth::Int64, bm::BartModel)
    bm.hypers.α*(1+depth)^(-bm.hypers.β)
end

# Probability ratio from a MH step from a GROW move ( no node to calculate the whole tree)
function get_log_prob_grow_leaf(depth::Int64, bm::BartModel)

    prob_selected_leaf_nonterminal::Float64 = get_prob_non_terminal(depth,bm) # Calculate here before to avoid calculate twice later
    return (log(prob_selected_leaf_nonterminal)+ 2*log(1-get_prob_non_terminal((depth+1),bm))) - log(1-prob_selected_leaf_nonterminal)

end

function get_log_prob_prune_node(depth::Int64, bm::BartModel)

    prob_selected_node_nonterminal::Float64 = get_prob_non_terminal(depth,bm) # Calculate here before to avoid calculate twice later

    return log(1-prob_selected_leaf_nonterminal) - (log(prob_selected_node_nonterminal) + 2*log(1-get_prob_non_terminal(depth+1,bm))) 
end