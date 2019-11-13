### ToDos:
## 7) scale in and outputs (with adjustment of the values afterwords)
## for better convergence
## 10) allow predicition
## mixture in python:
#mean,var,pi have the same shape(3,4).
# mean = tf.convert_to_tensor(np.arange(12.0).reshape(3,4))
# var = mean
# dist = tfd.Normal(loc=-1., scale=0.1)
#
# pi = tf.transpose(tf.ones_like(mean))
#
# mix = tfd.Mixture(cat = tfd.Categorical(probs=[pi/3,
#                                                pi/3,
#                                                pi/3]),
#                   components=[tfd.Normal(loc=mean,scale=var),
#                               tfd.Normal(loc=mean,scale=var),
#                               tfd.Normal(loc=mean,scale=var)]
# )

#'@title Fitting Deep Distributional Regression
#'
#'
#' @param list_of_formulae a named list of right hand side formulae,
#' one for each parameter of the distribution specified in \code{family};
#' set to \code{~ 1} if the parameter should be treated as constant.
#' Use the \code{s()}-notation from \code{mgcv} for specification of
#' non-linear structured effects and \code{d(..., model = ...)} for
#' deep learning predictors (separated by commas),
#' where \code{model} is the an integer
#' giving the index of the deep model in \code{list_of_deep_models} to
#' be used for the predictors
#' @param list_of_deep_models a list of (lists of) functions
#' specifying a keras model for each parameter of interest.
#' See the examples for more details.
#' @param family a character specifying the distribution. For information on 
#' possible distribution and parameters, see \code{\link{make_tfd_dist}} 
#' @param dist_fun a custom distribution applied to the last layer,
#' see \code{\link{make_tfd_dist}} for more details on how to construct
#' a custom distribution function.
#' @param variational logical value specifying whether or not to use
#' variational inference. If \code{TRUE}, details must be passed to
#' the via the ellipsis to the initialization function
#' (see \code{\link{deepregression_init}})
#' @param ... further arguments passed to the \code{deepRegression_init} function
#'
#' @import tensorflow tfprobability keras mgcv dplyr R6 reticulate Matrix
#' @export deepregression
#'
#' @examples
#' library(deepregression)
#'
#' data = data.frame(matrix(rnorm(10*100), c(100,10)))
#' colnames(data) <- c("x1","x2","x3","xa","xb","xc","xd","xe","xf","unused")
#' formula <- ~ 1 + d(x1,x2,x3) +
#' s(xa, sp = 1) + te(xe,xf) + x1
#'
#' deep_model <- function(x) x %>%
#' layer_dense(units = 128, activation = "relu", use_bias = FALSE) %>%
#' layer_dense(units = 64, activation = "relu") %>%
#' layer_dropout(rate = 0.2) %>%
#' layer_dense(units = 32, activation = "relu") %>%
#' layer_dropout(rate = 0.2) %>%
#' layer_dense(units = 8, activation = "relu") %>%
#' layer_dense(units = 1, activation = "linear")
#'
#' y <- rnorm(100)
#'
#' mod <- deepregression(list_of_formulae = list(loc = formula, scale = ~ 1),
#' data = data, validation_data = list(data, y), y = y,
#' list_of_deep_models = list(deep_model, NULL))
#'
#'mod %>% fit(epochs = 100)
#'mod %>% plot()
deepregression <- function(
  y,
  list_of_formulae,
  list_of_deep_models,
  family = c(
    "normal", "bernoulli", "bernoulli_prob", "beta", "betar",
    "cauchy", "chi2", "chi","exponential", "gamma_gamma",
    "gamma", "gammar", "gumbel", "half_cauchy", "half_normal", "horseshoe",
    "inverse_gamma", "inverse_gaussian", "laplace", "log_normal", "logistic",
    "negbinom", "pareto", "poisson", "poisson_lograte", "student_t",
    "student_t_ls", "truncated_normal", "uniform"
  ),
  train_together = FALSE,
  mean_regression = FALSE,
  data,
  # batch_size = NULL,
  # epochs = 10L,
  df = NULL,
  # defaultSp = 1,
  # defaultSmoothing = function(smoothTerm){
  #   smoothTerm$sp = defaultSp
  #   return(smoothTerm)
  # },
  defaultSmoothing = NULL,
  validation_data = NULL,
  validation_split = ifelse(is.null(validation_data), 0.2, 0),
  # verbose = getOption("keras.fit_verbose", default = 1),
  # shuffle = TRUE,
  # view_metrics = getOption("keras.view_metrics", default = "auto"),
  # class_weight = NULL,
  # sample_weight = NULL,
  dist_fun = NULL,
  learning_rate = 0.01,
  variational = FALSE,
  monitor_metric = list(),
  ...
)
{
  
  # check family
  family <- match.arg(family)
  # get column names of data
  varnames <- names(data)
  # number of observations
  n_obs <- nrow(data)


  # parse formulae
  parsed_formulae_contents <- lapply(list_of_formulae,
                                     get_contents,
                                     data = data,
                                     df = df,
                                     variable_names = varnames,
                                     defaultSmoothing = defaultSmoothing)

  this_OX <- lapply(parsed_formulae_contents, make_orthog, retcol = TRUE)

  # add intercept for parameters which are estimated as constant
  # replace_items = which(
  #   sapply(parsed_formulae_contents, is.null) &
  #     sapply(list_of_formulae, function(x) attr(terms.formula(x), "intercept")==1)
  # )
  # for(i in replace_items)
  #   parsed_formulae_contents[[i]] <- list(linterms = data.frame("(Intercept)"=rep(1, n_obs)),
  #                                         smoothterms = NULL,
  #                                         deepterms = NULL)

  # are parameters trained together?
  if(train_together)
  {

      # check if d-terms and corresponding covariates are in all
      # parameter

  }

  # extract constraints
  # TODO

  # get columns per term
  ncol_deep <- unlist(sapply(lapply(
    parsed_formulae_contents, "[[", "deepterms"), NCOL0))
  ncol_structured <- sapply(
    parsed_formulae_contents[!sapply(parsed_formulae_contents,is.null)],
    function(x){
      ncolsmooth <- 0
      if(!is.null(x[['smoothterms']]))
        ncolsmooth <- sum(sapply(x[['smoothterms']], function(st) NCOL(st$X)))

      return(NCOL(x[['linterms']]) + ncolsmooth)

    })
  # create structured layers
  list_structured <- lapply(1:length(parsed_formulae_contents), function(i)
                            get_layers_from_s(parsed_formulae_contents[[i]], i))

  pwr_input = NULL
  if(sum(ncol_deep)>0) pwr_input <- 1

  # initialize the model
  model <- deepregression_init(
    n_obs = n_obs,
    ncol_structured = ncol_structured,
    ncol_deep = ncol_deep,
    list_structured = list_structured,
    list_deep = list_of_deep_models,
    input_pwr = pwr_input,
    nr_params = length(list_structured),
    lss = TRUE,
    number_parameters_deep_together = as.numeric(train_together),
    family = family,
    variational = variational,
    dist_fun = dist_fun,
    kl_weight = 1 / n_obs,
    orthogX = this_OX,
    monitor_metric = monitor_metric,
    ...
    )


  # get covariates
  #
  # must be a list of the following form:
  # list(deep_part_param1, deep_part_param2, ..., deep_part_param_u,
  #      deep_struct_param1, deep_struct_param2, ..., deep_struct_param_r)
  input_cov <- make_cov(parsed_formulae_contents)
  # apply to validation data -> ?
  this_val_data <- validation_data
  this_val_split <- validation_split

  param_names <- names(parsed_formulae_contents)
  l_names_effets <- lapply(parsed_formulae_contents, get_names)
  ind_structterms <- lapply(parsed_formulae_contents, get_indices)

  if(!is.null(validation_data))
    validation_data[[1]] <- prepare_newdata(parsed_formulae_contents,
                                            validation_data[[1]],
                                            pred = TRUE)

  ret <- list(model = model,
              init_params =
                list(
                  input_cov = input_cov,
                  n_obs = n_obs,
                  y = y,
                  validation_split = validation_split,
                  validation_data = validation_data,
                  l_names_effets = l_names_effets,
                  parsed_formulae_contents = parsed_formulae_contents,
                  data = data,
                  ind_structterms = ind_structterms,
                  param_names = param_names,
                  ellipsis = list(...)
                ))

  class(ret) <- "deepregression"

  return(ret)

}

#' @title Initializing Deep Distributional Regression Models
#'
#'
#' @param n_obs number of observations
#' @param ncol_structured a vector of length #parameters
#' defining the number of variables used for each of the parameters.
#' If any of the parameters is not modelled using a structured part
#' the corresponding entry must be zero.
#' @param ncol_deep a vector of length #parameters
#' defining the number of variables used for each of the parameters.
#' If any of the parameters is not modelled using a deep part
#' the corresponding entry must be zero. If all parameters
#' are estimated by the same deep model, the first entry must be
#' non-zero while the others must be zero.
#' @param list_structured list of (non-linear) structured layers
#' (list length between 0 and number of parameters)
#' @param list_deep list of deep models to be used
#' (list length between 0 and number of parameters)
#' @param residual_projection logical; defines which type of
#' projection is used.
#' If \code{TRUE}, the deep part is estimated together with the structured part;
#' if \code{FALSE}, the deep part is fitted on the residuals of the structured part.
#'
#' @export deepregression_init
#'
deepregression_init <- function(
  n_obs,
  ncol_structured,
  ncol_deep,
  list_structured,
  list_deep,
  input_pwr=NULL,
  use_bias_in_structured = TRUE,
  nr_params = 2,
  lss = TRUE,
  number_parameters_deep_together = 0,
  family,
  dist_fun = NULL,
  variational = TRUE,
  weights = NULL,
  learning_rate = 0.01,
  optimizer = optimizer_adam(lr = learning_rate),
  monitor_metric = list(),
  mean_field = posterior_mean_field,
  prior = prior_trainable,
  orthog_fun = orthog,
  orthogX = NULL,
  inject_after_layer = rep(-1, length(list_deep)),
  residual_projection = FALSE,
  kl_weight = 1 / n_obs)
{

  # check distribution wrt to specified parameters
  nrparams_dist <- make_tfd_dist(family, return_nrparams = TRUE)
  if(nrparams_dist < nr_params)
  {
    warning("More formulae specified than parameters available.",
            " Will only use ", nrparams_dist, " formula(e).") 
    nr_params <- nrparams_dist
    ncol_deep <- ncol_deep[1:nr_params]
    ncol_structured <- ncol_structured[1:nr_params]
  }
  # check injection
  if(length(inject_after_layer) > nr_params)
    stop("Can't have more injections than parameters.")
  if(any(sapply(inject_after_layer, function(x) x%%1!=0)))
    stop("inject_after_layer must be a positive / negative integer")

  # define the input layers
  inputs_deep <- lapply(ncol_deep, function(nc){
    if(nc==0) return(NULL) else
      layer_input(shape = list(nc))
    })
  inputs_struct <- lapply(1:length(ncol_structured), function(i){
    nc = ncol_structured[i]
    if(nc==0) return(NULL) else
      if(!is.null(list_structured[[i]]) & nc > 1)
        # nc>1 will cause problems when implementing ridge/lasso
        layer_input(shape = list(1,nc)) else
          layer_input(shape = list(nc))
  })
  if(!is.null(input_pwr)){
    pwr = layer_input(shape = 1)
  }else{
    pwr = NULL
  }

  if(!is.null(orthogX)){
    ox <- lapply(orthogX, function(x) if(is.null(x)) return(NULL) else
      return(layer_input(shape = list(x))))
  }

  # define structured predictor
  structured_parts <- lapply(1:length(inputs_struct),
                             function(i){
                               if(is.null(inputs_struct[[i]]))
                               {
                                 return(NULL)
                               }else{
                                 if(is.null(list_structured[[i]]))
                                 {
                                   return(inputs_struct[[i]] %>%
                                            layer_dense(units = 1, activation = "linear",
                                                        use_bias = use_bias_in_structured,
                                                        name = paste0("structured_linear_",i))
                                   )
                                 }else{
                                   this_layer <- list_structured[[i]]
                                   return(inputs_struct[[i]] %>% this_layer)
                                 }
                               }
                             })


  # split deep parts in two parts, where
  # the first part is used in the orthogonalization
  # and the second is put back on top of the first
  # after orthogonalization

  if(number_parameters_deep_together == 0 &
     length(inputs_deep) != length(list_deep) & any(!sapply(inputs_deep, is.null)))
    stop(paste0("If paramters of distribution are not trained together, ",
         "a deep model must be provided for each parameter."))
  deep_split <- lapply(1:length(inputs_deep),
                       function(i)
                         if(is.null(inputs_deep[[i]])) return(NULL) else
                           split_model(list_deep[[i]], inject_after_layer[i])
  )

  list_deep <- lapply(deep_split, "[[", 1)
  list_deep_ontop <- lapply(deep_split, "[[", 2)

  # define deep predictor
  deep_parts <- lapply(1:length(inputs_deep), function(i)
    if(is.null(inputs_deep[[i]])) return(NULL) else
      list_deep[[i]](inputs_deep[[i]]))

  # split deep parts if trained together
  if(number_parameters_deep_together){

    stop("Training together must be rewritten for new version.")

    if(length(deep_parts) > 1){

      stop("Training deep parts together for more than one deep model not supported yet.")

    }

    # function for split deep model parts
    split_fun <- function(x)
      tf$split(x, num_or_size_splits =
                 as.integer(number_parameters_deep_together),
               axis = as.integer(1))

    deep_parts <- layer_lambda(object = deep_parts[[1]],
                               f = split_fun)

  }

  ############################################################
  ################# Apply Orthogonalization ##################

  # create final linear predictor per distribution parameter
  # -> depending on the presence of a deep or structured part
  # the corresponding part is returned. If both are present
  # the deep part is projected into the orthogonal space of the
  # structured part
  list_pred_param <- lapply(1:nr_params,
                            function(i){
                              if(is.null(deep_parts[[i]])){
                                return(structured_parts[[i]])
                              }else if(is.null(structured_parts[[i]])){
                                return(deep_parts[[i]] %>%
                                         list_deep_ontop[[i]])
                              }else{

                                if(is.null(ox[[i]])){
                                  return(layer_add(
                                    list(
                                      list_deep_ontop[[i]](deep_parts[[i]]),
                                         structured_parts[[i]]
                                    )))
                                }else{
                                  return(
                                    layer_add(
                                      list(
                                        list_deep_ontop[[i]](
                                          orthog_fun(deep_parts[[i]],
                                                     ox[[i]],
                                                     pwr)
                                        ),
                                        structured_parts[[i]]
                                        )
                                    )
                                  )
                                }
                              }
                            })


  # concatenate predictors
  # -> just to split them later again?
  if(length(list_pred_param) > 1)
    preds <- layer_concatenate(list_pred_param) else
      preds <- list_pred_param[[1]]

  ############################################################
  ### Define Distribution Layer and Variational Inference ####

   
  
  # define the distribution function applied in the last layer

  if(lss){

    # special families needing transformations
    if(family %in% c("betar", "gammar", "negbinom")){
      
      trafo_fun <- family_trafo_funs(family)
      predsTrafo <- layer_lambda(object = preds, f = trafo_fun)
      preds <- layer_concatenate(predsTrafo)
      
    }
    
    # apply the transformation for each parameter
    # and put in the right place of the distribution
    if(is.null(dist_fun))
      dist_fun <- make_tfd_dist(family)

    # make model variational and output distribution
    if(variational){

      out <- preds %>%
        layer_dense_variational(
          units = length(list_pred_parameters),
          make_posterior_fn = mean_field,
          make_prior_fn = prior,
          kl_weight = kl_weight
        ) %>%
        layer_distribution_lambda(dist_fun)

    }else{

      out <- preds %>%
        layer_distribution_lambda(dist_fun)

    }

  }else{
    # no location scale and shape model
    # -> just modelling the mean

    # Use the specified distribution
    # check for mean in distribution
    # model remaining parameter with constant
    stop("Not implemented yet.")

  }

  ############################################################
  ################# Define and Compile Model #################

  # define all inputs
  inputList <- unname(c(
    inputs_deep[!sapply(inputs_deep, is.null)],
    inputs_struct[!sapply(inputs_struct, is.null)],
    pwr,
    ox[!sapply(ox, is.null)])
  )
  # the final model is defined by its inputs
  # and outputs
  model <- keras_model(inputs = inputList,
                       outputs = out)

  # define weights to be equal to 1 if not given
  if(is.null(weights)) weights <- rep(1, n_obs)

  # the negative log-likelihood is given by the negative weighted
  # log probability of the model
  negloglik <- function(y, model) - weights * (model %>% tfd_log_prob(y))

  # compile the model using the defined optimizer,
  # the negative log-likelihood as loss funciton
  # and the defined monitoring metrics as metrics
  model %>% compile(optimizer = optimizer,
                    loss = negloglik,
                    metrics = monitor_metric)

  # model$set("public", "get_linear_weights", function() {
  #   warning("get_linear_weights is only working in special cases.")
  #   return(self$)
  # })

  return(model)

}

