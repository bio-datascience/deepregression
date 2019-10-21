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
  if(all(this_ind$type!="
         smooth")) return("No smooth effects. Nothing to plot.")
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
           value = object$init_params$data[,nam],
           design_mat = BX,
           coef = phi[this_ind_this_w,],
           partial_effect = BX%*%phi[this_ind_this_w,])
    if(plot) plot(partial_effect ~ value,
                  data = plotData[[w]],
                  main = paste0("s(", nam, ")"),
                  xlab = nam,
                  ylab = "partial effect",
                  ...)

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
    deepregression.predict(object, apply_fun=apply_fun, ...)
  )
}



#' @method fit deepregression
#' @export
#' @rdname methodDR
#'
fit.deepregression <- function(
  object,
  ...
)
{

  args <- list(...)
  args <- append(args,
                 list(object = object$model,
                      x = prepare_newdata(object$init_params$parsed_formulae_contents,
                                          object$init_params$data,
                                          pred = FALSE),
                      y = object$init_params$y,
                      validation_split = object$init_params$validation_split,
                      validation_data = object$init_params$validation_data))
  args <- append(args, object$init_params$ellipsis)

  do.call(keras:::fit.keras.engine.training.Model,
          args)
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
