function StatsBase.fit(BartModel,X_train::Matrix{Float64}, y_train::Vector{Float64},numcut = 100, usequant = true,mcmc = MCMC(); hyperargs...)
   
    # this is initialised only once and is not udpated
    bart_model = BartModel(X_train,y_train,mcmc,numcut,usequant;hyperargs...)

    # UButuak ibe
    bart_state = StandardBartState(bart_model)

    # return bart_state
    
    # Creating a element to create all the BART state
    # all_bart_states = Vector{StandardBartState}(undef,bart_model.mcmc.niter) # Just need this later if I want to investigate the complete chain 
    n_post = bart_model.mcmc.niter-bart_model.mcmc.nburn
    bart_trees_post = Vector{BartEnsemble}(undef,n_post)
    sigmas_post = zeros(Float64,n_post)
    fhat_post = zeros(Float64,n_post,bart_model.td.n)
    post_iter = 1
    
    for i in 1:bart_model.mcmc.niter
        draw_trees!(bart_state,bart_model)
        draw_s!(bart_state,bart_model)
        draw_σ!(bart_state,bart_model)
        
        if i>bart_model.mcmc.nburn
            bart_trees_post[post_iter] = deepcopy(bart_state.ensemble)
            sigmas_post[post_iter] = bart_state.σ  
            fhat_post[post_iter,:] = bart_state.fhat

            post_iter+=1
        end

        progress_bar(i,bart_model.mcmc.niter)

    end

    return BartChain(bart_state,bart_model,bart_trees_post,sigmas_post,fhat_post,mean(fhat_post,dims = 1)[1,:])

end

