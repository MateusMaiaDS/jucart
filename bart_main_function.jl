function StatsBase.fit(BartModel,X_train::Matrix{Float64}, y_train::Vector{Float64},numcut = 100, usequant = true,mcmc = MCMC(); hyperargs...)
   
    # this is initialised only once and is not udpated
    bart_model = BartModel(X_train,y_train,mcmc,numcut,usequant;hyperargs...)

    # UButuak ibe
    bart_state = StandardBartState(bart_model)

    # return bart_state
    
    # Creating a element to create all the BART state
    all_bart_states = Vector{StandardBartState}(undef,bart_model.mcmc.niter)
    fhat_post = zeros(Float64,bart_model.mcmc.niter-bart_model.mcmc.nburn,bart_model.td.n)
    post_iter = 1
    for i in 1:1
        draw_trees!(bart_state,bart_model)
        draw_s!(bart_state,bart_model)
        draw_σ!(bart_state,bart_model)
        all_bart_states[i] = deepcopy(bart_state)
        
        if i>bart_model.mcmc.nburn
            fhat_post[post_iter,:] = bart_state.fhat
            post_iter+=1
        end

        progress_bar(i,bart_model.mcmc.niter)

    end

    return fhat_post

end