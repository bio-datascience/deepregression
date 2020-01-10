#' @title Generic functions for deepregression models
#'
#' @param object deepregression object
#' @param which which effect to plot, default selects all.
#' @param which_param integer of length 1.
#' Corresponds to the distribution parameter for 
#' which the effects should be plotted.
#' @param use_posterior logical; if \code{TRUE} it is assumed that
#' the strucuted_nonlinear layer has stored a list of length two
#' as weights, where the first entry is a vector of mean and sd
#' for each network weight. The sd is transformed using the \code{exp} function.
#' The plot then shows the mean curve +- 2 times sd.
#' @param ... further arguments, passed to fit, plot or predict function
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
  use_posterior = FALSE,
  ... # passed to plot function
)
{
  this_ind <- object$init_params$ind_structterms[[which_param]]
  if(all(this_ind$type!="smooth")) return("No smooth effects. Nothing to plot.")
  if(is.null(which)) which <- 1:length(which(this_ind$type=="smooth"))
  plus_number_lin_eff <- sum(this_ind$type=="lin")

  plotData <- vector("list", length(which))
  org_feature_names <- 
    names(object$init_params$l_names_effets[[which_param]][["smoothterms"]])
  phi <- object$model$get_layer(paste0("structured_nonlinear_",
                                       which_param))$get_weights()
  if(length(phi)>1){
    if(use_posterior){
      phi <- matrix(phi[[1]], ncol=2, byrow=F)
    }else{
      phi <- as.matrix(phi[[2]], ncol=1)
    }
  }else{
    phi <- phi[[1]]
  }
  
  for(w in which){

    nam <- org_feature_names[w]
    this_ind_this_w <- do.call("Map",
                               c(":", as.list(this_ind[w+plus_number_lin_eff,
                                                       c("start","end")])))[[1]]
    BX <- 
      object$init_params$parsed_formulae_contents[[
        which_param]]$smoothterms[[nam]]$X
    if(use_posterior){
      
      # get the correct index as each coefficient has now mean and sd
      phi_mean <- phi[this_ind_this_w,1]
      phi_sd <- log(exp(log(expm1(1)) + phi[this_ind_this_w,2])+1)
      plotData[[w]] <- 
        list(org_feature_names = nam,
             value = unlist(object$init_params$data[strsplit(nam,",")[[1]]]),
             design_mat = BX,
             coef = phi[this_ind_this_w,],
             mean_partial_effect = BX%*%phi_mean,
             sd_partial_effect = sqrt(diag(BX%*%diag(phi_sd^2)%*%t(BX))))
    }else{
      plotData[[w]] <-
        list(org_feature_name = nam,
             value = unlist(object$init_params$data[strsplit(nam,",")[[1]]]),
             design_mat = BX,
             coef = phi[this_ind_this_w,],
             partial_effect = BX%*%phi[this_ind_this_w,])
    }
    if(plot){
      nrcols <- NCOL(plotData[[w]]$value)
      if(nrcols==1)
      {
        if(use_posterior){
          plot(mean_partial_effect[order(value)] ~ sort(value),
               data = plotData[[w]],
               main = paste0("s(", nam, ")"),
               xlab = nam,
               ylab = "partial effect",
               ylim = c(min(mean_partial_effect - 2*sd_partial_effect),
                        max(mean_partial_effect + 2*sd_partial_effect)),
               ...)
          with(plotData[[w]], {
            points((mean_partial_effect + 2 * sd_partial_effect)[order(value)] ~
                     sort(value), type="l", lty=2)
            points((mean_partial_effect - 2 * sd_partial_effect)[order(value)] ~
                     sort(value), type="l", lty=2)
          })
        }else{
          plot(partial_effect[order(value)] ~ sort(value),
               data = plotData[[w]],
               main = paste0("s(", nam, ")"),
               xlab = nam,
               ylab = "partial effect",
               ...)
        }
      }else if(nrcols==2){
        # this_data = cbind(plotData[[w]]$value,partial_effect=plotData[[w]]
        # $partial_effect)
        # image(plotData[[w]]$value[,1],
        #               plotData[[w]]$value[,2],
        #               plotData[[w]]$partial_effect,
        #               ...,
        #               xlab = names(plotData[[w]]$value)[1],
        #               ylab = names(plotData[[w]]$value)[2],
        #               zlab = "partial effect",
        #               main = paste0("te(", nam, ")")
        # )
        warning("Plotting of effects with ", nrcols, " 
                covariate inputs not supported yet.")
      }else{
        warning("Plotting of effects with ", nrcols, 
                " covariate inputs not supported.")
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
  data,
  pred=FALSE
)
{
  newdata_processed <- prepare_newdata(
    object$init_params$parsed_formulae_contents,
    data, pred=pred)
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
    yhat <- object$model(unname(object$init_params$input_cov))
  }else{
    # preprocess data
    if(is.data.frame(newdata)) newdata <- as.list(newdata)
    newdata_processed <- prepare_data(object, newdata, pred=TRUE)
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
#' @param ... further arguments passed to 
#' \code{keras:::fit.keras.engine.training.Model}
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
  input_list_model <- 
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
  args <- append(args, 
                 input_list_model[!names(input_list_model) %in% 
                                    names(args)])
  if(length(object$init_params$ellipsis)>0)
    args <- append(args, 
                   object$init_params$ellipsis[
                     !names(object$init_params$ellipsis) %in% names(args)])

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
  object,
  variational = FALSE
)
{
  nrparams <- length(object$init_params$parsed_formulae_contents)
  layer_names <- sapply(object$model$layers, "[[", "name")
  lret <- vector("list", nrparams)
  names(lret) <- object$init_params$param_names
  for(i in 1:nrparams){
    sl <- paste0("structured_linear_",i)
    slas <- paste0("structured_lasso_",i)
    snl <- paste0("structured_nonlinear_",i)
    lret[[i]] <- list(structured_linear = NULL,
                      structured_lasso = NULL,
                      structured_nonlinear = NULL)

    lret[[i]]$structured_linear <-
      if(sl %in% layer_names)
        object$model$get_layer(sl)$get_weights()[[1]] else
          NULL
    lret[[i]]$structured_lasso <-
      if(slas %in% layer_names)
        object$model$get_layer(slas)$get_weights()[[1]] else
          NULL
    if(snl %in% layer_names){
      cf <- object$model$get_layer(snl)$get_weights()
      if(length(cf)==2 & variational){
        lret[[i]]$structured_nonlinear <-  cf[[1]]
      }else{
        lret[[i]]$structured_nonlinear <- cf[[length(cf)]]
      }
    }else{
      lret[[i]]$structured_nonlinear <- NULL
    }

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
#' @param ... further arguments passed to 
#' \code{keras:::fit.keras.engine.training.Model}
#' @param object deepregression object
#' @param verbose whether to print training in each fold
#' @param patience number of patience for early stopping
#' @param plot whether to plot the resulting losses in each fold
#' @param printfolds whether to print the current fold
#' @param mylapply lapply function to be used; defaults to \code{lapply}
#' @param cv_folds see \code{deepregression}
#' @param stop_if_nan logical; whether to stop CV if NaN values occur
#' @export
#' @rdname methodDR
#' 
#' @return Returns an object \code{drCV}, a list, one list element for each fold
#' containing the model fit and the \code{weighthistory}.
#' 
#'
#' 
cv <- function(
  object,
  verbose = FALSE, 
  patience = 20,
  plot = TRUE,
  print_folds = TRUE,
  cv_folds = NULL,
  stop_if_nan = TRUE,
  mylapply = lapply,
  ...
)
{
  
  if(is.null(cv_folds)){ 
    cv_folds <- object$init_params$cv_folds
  }else if(!is.list(cv_folds) & is.numeric(cv_folds)){
    cv_folds <- make_cv_list_simple(
      data_size = NROW(object$init_params$data[[1]]), 
      cv_folds)
  }else{
    stop("Wrong format for cv_folds.")
  }
  if(is.null(cv_folds)){
    warning("No folds for CV given, using k = 10.\n")
    cv_folds <- make_cv_list_simple(
      data_size = NROW(object$init_params$data[[1]]), 10)
  }
  nrfolds <- length(cv_folds)
  old_weights <- object$model$get_weights()
  
  if(print_folds) folds_iter <- 1
  
  # subset fun
  if(NCOL(object$init_params$y)==1)
    subset_fun <- function(y,ind) y[ind] else
      subset_fun <- function(y,ind) subset_array(y,ind)
  
  res <- mylapply(cv_folds, function(this_fold){
  
    if(print_folds) cat("Fitting Fold ", folds_iter, " ... ")
    st1 <- Sys.time()
    
    # does not work?
    # this_mod <- clone_model(object$model)
    this_mod <- object$model
    
    train_ind <- this_fold[[1]]
    test_ind <- this_fold[[2]]
    
    # data
    if(is.data.frame(object$init_params$data)){
      train_data <- object$init_params$data[train_ind,, drop=FALSE] 
      test_data <- object$init_params$data[test_ind,,drop=FALSE]
    }else if(class(object$init_params$data)=="list"){
      train_data <- lapply(object$init_params$data, function(x) 
        subset_array(x, train_ind))
      test_data <- lapply(object$init_params$data, function(x) 
        subset_array(x, test_ind))
    }else{
      stop("Invalid input format for CV.")
    }
    
    # make callbacks 
    this_callbacks <- list()
    weighthistory <- WeightHistory$new()
    this_callbacks <- append(this_callbacks, weighthistory)
    
    args <- list(...)
    args <- append(args,
                   list(object = this_mod,
                        x = prepare_newdata(
                          object$init_params$parsed_formulae_contents,
                          train_data,
                          pred = FALSE,
                          index = train_ind),
                        y = subset_fun(object$init_params$y,train_ind),
                        validation_split = NULL,
                        validation_data = list(
                          prepare_newdata(
                            object$init_params$parsed_formulae_contents,
                            test_data,
                            pred = FALSE,
                            index = test_ind),
                          subset_fun(object$init_params$y,test_ind)
                        ),
                        callbacks = this_callbacks,
                        verbose = verbose,
                        view_metrics = FALSE
                   )
    )
    args <- append(args, object$init_params$ellipsis)
    
    ret <- do.call(fit_fun, args)
    ret$weighthistory <- weighthistory$weights_last_layer
    
    if(stop_if_nan && any(is.nan(ret$metrics$validloss)))
      stop("Fold ", folds_iter, " with NaN's in ")
  
    if(print_folds) folds_iter <<- folds_iter + 1
    
    this_mod$set_weights(old_weights)
    td <- Sys.time()-st1
    if(print_folds) cat("\nDone in", as.numeric(td), "", attr(td,"units"), "\n")
    
    return(ret)
    
  })
  
  class(res) <- c("drCV","list")
  
  if(plot) try(plot(res))
  
  object$model$set_weights(old_weights)
  
  invisible(return(res))
  
}

#' mean of model fit
#' 
#' @method mean deepregression
#' @export
#' @rdname methodDR
#'
mean.deepregression <- function(
  object,
  data,
  ...
)
{
  predict.deepregression(object, newdata = data, apply_fun = tfd_mean, ...)
}

#' standard deviation of model fit
#' 
#' @method sd deepregression
#' @export
#' @rdname methodDR
#'
sd.deepregression <- function(
  object,
  data,
  ...
)
{
  predict.deepregression(object, newdata = data, apply_fun = tfd_stddev, ...)
}

#' quantile of fitted values
#' 
#' @method quantile deepregression
#' @export
#' @rdname methodDR
#'
quantile.deepregression <- function(
  object,
  data,
  value,
  ...
)
{
  predict.deepregression(object, 
                         newdata = data, 
                         apply_fun = function(x) tfd_quantile(x, value=value),
                         ...)
}