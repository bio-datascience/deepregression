#' @title Generic functions for deepregression models
#'
#' @param object deepregression object
#' @param which which effect to plot, default selects all.
#' @param which_param integer of length 1.
#' Corresponds to the distribution parameter for which the effects should be plotted.
#' @param ... further arguments, passed to fit, plot or predict function
#' @param apply_fun function to apply to distribution, default is \code{tfd_mean}.
#' @param newdata new data for prediction, defaults to \code{NULL}
#' @param convert_fun function which converts the tensor outputted by predict function
#'
#' @method plot deepregression
#' @export
#' @rdname methodDR
#'
plot.deepregression <- function(
  object,
  which = NULL,
  # which of the nonlinear structured effects
  which_param = 1, # for which parameter
  plot = TRUE,
  ... # passed to plot function
)
{
  this_ind <- object$init_params$ind_structterms[[which_param]]
  if(all(this_ind$type!="smooth")) return("No smooth effects. Nothing to plot.")
  if(is.null(which)) which <- 1:length(which(this_ind$type=="smooth"))
  plus_number_lin_eff <- sum(this_ind$type=="lin")

  plotData <- vector("list", length(which))
  org_feature_names <- names(object$init_params$l_names_effets[[which_param]][["smoothterms"]])
  phi <- object$model$get_layer(paste0("structured_nonlinear_",
                                       which_param))$get_weights()[[1]]

  for(w in which){

    nam <- org_feature_names[w]
    this_ind_this_w <- do.call("Map",
                               c(":", as.list(this_ind[w+plus_number_lin_eff,
                                                       c("start","end")])))[[1]]
    BX <- object$init_params$parsed_formulae_contents[[which_param]]$smoothterms[[nam]]$X
    plotData[[w]] <-
      list(org_feature_name = nam,
           value = object$init_params$data[,strsplit(nam,",")[[1]]],
           design_mat = BX,
           coef = phi[this_ind_this_w,],
           partial_effect = BX%*%phi[this_ind_this_w,])
    if(plot){
      nrcols <- NCOL(plotData[[w]]$value)
      if(nrcols==1)
      {
        plot(partial_effect ~ value,
             data = plotData[[w]],
             main = paste0("s(", nam, ")"),
             xlab = nam,
             ylab = "partial effect",
             ...)
      }else if(nrcols==2){
        # this_data = cbind(plotData[[w]]$value,partial_effect=plotData[[w]]$partial_effect)
        # image(plotData[[w]]$value[,1],
        #               plotData[[w]]$value[,2],
        #               plotData[[w]]$partial_effect,
        #               ...,
        #               xlab = names(plotData[[w]]$value)[1],
        #               ylab = names(plotData[[w]]$value)[2],
        #               zlab = "partial effect",
        #               main = paste0("te(", nam, ")")
        # )
        warning("Plotting of effects with ", nrcols, " covariate inputs not supported yet.")
      }else{
        warning("Plotting of effects with ", nrcols, " covariate inputs not supported.")
      }
    }
  }

  invisible(plotData)
}


#' @export
#' @rdname methodDR
#'
prepare_data <- function(
  object,
  data
)
{
  newdata_processed <- prepare_newdata(object$init_params$parsed_formulae_contents,
                                       data)
  return(newdata_processed)
}


#' @method predict deepregression
#' @export
#' @rdname methodDR
#'
predict.deepregression <- function(
  object,
  newdata = NULL,
  apply_fun = tfd_mean,
  convert_fun = as.matrix,
  ...
)
{

  if(is.null(newdata)){
    yhat <- object$model(c(unname(object$init_params$input_cov),
                           list(rep(0, object$init_params$n_obs))))
  }else{
    # preprocess data
    newdata_processed <- prepare_data(object, newdata)
    yhat <- object$model(newdata_processed)
  }


  if(!is.null(apply_fun))
    return(convert_fun(apply_fun(yhat))) else
      return(convert_fun(yhat))

}

#' @method fitted deepregression
#' @export
#' @rdname methodDR
#'
fitted.deepregression <- function(
  object, apply_fun = tfd_mean, ...
)
{
  return(
    predict.deepregression(object, apply_fun=apply_fun, ...)
  )
}



#' @method fit deepregression
#' @param ... further arguments passed to \code{keras:::fit.keras.engine.training.Model}
#' such as \code{verbose} (logical or 0/1), \code{view_metrics} (logical or )
#' @export
#' @rdname methodDR
#'
fit.deepregression <- function(
  object,
  early_stopping = FALSE,
  verbose = FALSE, 
  view_metrics = TRUE,
  patience = 20,
  save_weights = FALSE,
  ...
)
{

  this_callbacks <- list()
  
  # make callbacks 
  if(save_weights){
    weighthistory <- WeightHistory$new()
    this_callbacks <- append(this_callbacks, weighthistory)
  }
  if(early_stopping)
    this_callbacks <- append(this_callbacks, 
                             callback_early_stopping(patience = patience))
  # if(monitor_weights){
  #   # object$history <- WeightHistory$new()
  #   weight_callback <- callback_lambda(
  #     on_epoch_begin = function(epoch, logs) 
  #       coef_prior_to_epoch <<- unlist(coef(object)),
  #     on_epoch_end = function(epoch, logs) 
  #       print(sum(abs(coef_prior_to_epoch-unlist(coef(object)))))
  #     )
  #   this_callbacks <- append(this_callbacks, weight_callback)
  # }
  
  args <- list(...)
  args <- append(args,
                 list(object = object$model,
                      x = prepare_newdata(object$init_params$parsed_formulae_contents,
                                          object$init_params$data,
                                          pred = FALSE),
                      y = object$init_params$y,
                      validation_split = object$init_params$validation_split,
                      validation_data = object$init_params$validation_data,
                      callbacks = this_callbacks,
                      verbose = verbose,
                      view_metrics = view_metrics
                 )
  )
  args <- append(args, object$init_params$ellipsis)

  ret <- do.call(fit_fun,
                 args)
  if(save_weights) ret$weighthistory <- weighthistory$weights_last_layer
  invisible(ret)
}

#' @method coef deepregression
#' @export
#' @rdname methodDR
#'
coef.deepregression <- function(
  object
)
{
  nrparams <- length(object$init_params$parsed_formulae_contents)
  layer_names <- sapply(object$model$layers, "[[", "name")
  lret <- vector("list", nrparams)
  names(lret) <- object$init_params$param_names
  for(i in 1:nrparams){
    sl <- paste0("structured_linear_",i)
    snl <- paste0("structured_nonlinear_",i)
    lret[[i]] <- list(structured_linear = NULL,
                      structured_nonlinear = NULL)

    lret[[i]]$structured_linear <-
      if(sl %in% layer_names)
        object$model$get_layer(sl)$get_weights()[[1]] else
          NULL
    lret[[i]]$structured_nonlinear <-
      if(snl %in% layer_names)
        object$model$get_layer(snl)$get_weights()[[1]] else
          NULL

  }
  return(lret)

}

#' @method print deepregression
#' @export
#' @rdname methodDR
#'
print.deepregression <- function(
  object
)
{
  print(object$model)
}

#' @title Cross-validation for deepgression objects
#' @param ... further arguments passed to \code{keras:::fit.keras.engine.training.Model}
#' @param object deepregression object
#' @param verbose whether to print training in each fold
#' @param patience number of patience for early stopping
#' @param plot whether to plot the resulting losses in each fold
#' @param printfolds whether to print the current fold
#' @param mylapply lapply function to be used; defaults to \code{lapply}
#' @export
#' @rdname methodDR
#'
#' 
cv <- function(
  object,
  verbose = FALSE, 
  patience = 20,
  plot = TRUE,
  print_folds = TRUE,
  cv_folds = NULL,
  mylapply = lapply,
  ...
)
{
  
  cv_folds <- object$init_params$cv_folds
  if(is.null(cv_folds)){
    warning("No folds for CV given, using k = 10.\n")
    cv_folds <- make_cv_list_simple(data_size=nrow(object$init_params$data), 10)
  }
  nrfolds <- length(cv_folds)
  old_weights <- object$model$get_weights()
  
  if(print_folds) folds_iter <- 1
  
  res <- mylapply(cv_folds, function(this_fold){
  
    cat("Fitting Fold ", folds_iter, " ... ")
    
    # does not work?
    # this_mod <- clone_model(object$model)
    this_mod <- object$model
    
    train_ind <- this_fold[[1]]
    test_ind <- this_fold[[2]]
    
    # make callbacks 
    this_callbacks <- list()
    weighthistory <- WeightHistory$new()
    this_callbacks <- append(this_callbacks, weighthistory)
    
    args <- list(...)
    args <- append(args,
                   list(object = this_mod,
                        x = prepare_newdata(object$init_params$parsed_formulae_contents,
                                            object$init_params$data[train_ind,,drop=FALSE],
                                            pred = FALSE,
                                            index = train_ind),
                        y = object$init_params$y[train_ind],
                        validation_split = NULL,
                        validation_data = list(
                          prepare_newdata(object$init_params$parsed_formulae_contents,
                                          object$init_params$data[test_ind,,drop=FALSE],
                                          pred = TRUE,
                                          index = test_ind),
                          object$init_params$y[test_ind]
                        ),
                        callbacks = this_callbacks,
                        verbose = verbose,
                        view_metrics = FALSE
                   )
    )
    args <- append(args, object$init_params$ellipsis)
    
    ret <- do.call(fit_fun, args)
    ret$weighthistory <- weighthistory$weights_last_layer
  
    if(print_folds) folds_iter <<- folds_iter + 1
    
    this_mod$set_weights(old_weights)
    cat("\nDone.\n")
    
    return(ret)
    
  })
  
  if(plot) plot_cv_result(res)
  
  object$model$set_weights(old_weights)
  
  return(res)
  
}