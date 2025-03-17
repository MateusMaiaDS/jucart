function StatsBase.fit(BartModel,X_train::Matrix{Float64}, y_train::Vector{Float64},mcmc = MCMC(); hyperargs...)

    bart_model = BartModel(X_train,y_train,mcmc;hyperargs)
    