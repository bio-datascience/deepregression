#'@title Fitting Deep Distributional Regression
#'
#' @param y response variable
#' @param list_of_formulae a named list of right hand side formulae,
#' one for each parameter of the distribution specified in \code{family};
#' set to \code{~ 1} if the parameter should be treated as constant.
#' Use the \code{s()}-notation from \code{mgcv} for specification of
#' non-linear structured effects and \code{d(...)} for
#' deep learning predictors (predictors in brackets are separated by commas),
#' where \code{d} can be replaced by an name name of the names in 
#' \code{list_of_deep_models}, e.g., \code{~ 1 + s(x) + my_deep_mod(a,b,c)},
#' where my_deep_mod is the name of the neural net specified in 
#' \code{list_of_deep_models} and \code{a,b,c} are features modeled via
#' this network.
#' @param list_of_deep_models a named list of functions
#' specifying a keras model.
#' See the examples for more details.
#' @param family a character specifying the distribution. For information on 
#' possible distribution and parameters, see \code{\link{make_tfd_dist}} 
#' @param train_together logical; whether or not to train all parameters in 
#' one deep network.
#' @param data data.frame or named list with input features
#' @param df degrees of freedom for all non-linear structural terms
#' @param lambda_lasso smoothing parameter for lasso regression; 
#' can be combined with ridge
#' @param lambda_ridge smoothing parameter for ridge regression; 
#' can be combined with lasso
#' @param defaultSmoothing function applied to all s-terms, per default (NULL)
#' the minimum df of all possible terms is used.
#' @param cv_folds a list of lists, each list element has two elements, one for
#' training indices and one for testing indices; 
#' if a single integer number is given, 
#' a simple k-fold cross-validation is defined, where k is the supplied number.
#' @param validation_data data for validation during training.
#' @param validation_spit percentage of training data used for validation. 
#' Per default 0.2.
#' @param dist_fun a custom distribution applied to the last layer,
#' see \code{\link{make_tfd_dist}} for more details on how to construct
#' a custom distribution function.
#' @param learning_rate learning rate for optimizer
#' @param optimizer optimzer used. Per default ADAM.
#' @param variational logical value specifying whether or not to use
#' variational inference. If \code{TRUE}, details must be passed to
#' the via the ellipsis to the initialization function
#' (see \code{\link{deepregression_init}})
#' @param monitor_metric Further metrics to monitor
#' @param posterior_fun function defining the posterior function for the variational
#' verison of the network
#' @param prior_fun function defining the prior function for the variational
#' verison of the network
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
#' s(xa, sp = 1) + x1
#'
#' deep_model <- function(x) x %>%
#' layer_dense(units = 32, activation = "relu", use_bias = FALSE) %>%
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
#'
deepregression <- function(
  y,
  list_of_formulae,
  list_of_deep_models,
  family = c(
    "normal", "bernoulli", "bernoulli_prob", "beta", "betar",
    "cauchy", "chi2", "chi","exponential", "gamma_gamma",
    "gamma", "gammar", "gumbel", "half_cauchy", "half_normal", "horseshoe",
    "inverse_gamma", "inverse_gaussian", "laplace", "log_normal", "logistic",
    "multinomial", "multinoulli", "negbinom", "pareto", "poisson", 
    "poisson_lograte", "student_t",
    "student_t_ls", "truncated_normal", "uniform", "zip"
  ),
  train_together = FALSE,
  data,
  # batch_size = NULL,
  # epochs = 10L,
  df = NULL,
  lambda_lasso = NULL,
  lambda_ridge = NULL,
  # defaultSp = 1,
  # defaultSmoothing = function(smoothTerm){
  #   smoothTerm$sp = defaultSp
  #   return(smoothTerm)
  # },
  defaultSmoothing = NULL,
  cv_folds = NULL,
  validation_data = NULL,
  validation_split = ifelse(is.null(validation_data) & is.null(cv_folds), 0.2, 0),
  dist_fun = NULL,
  learning_rate = 0.01,
  optimizer = optimizer_adam(lr = learning_rate),
  variational = FALSE,
  monitor_metric = list(),
  seed = 1991-5-4,
  mixture_dist = 0,
  split_fun = split_model,
  posterior_fun = posterior_mean_field,
  prior_fun = prior_trainable,
  null.space.penalty = variational,
  ind_fun = function(x) tfd_independent(x),
  extend_output_dim = 0,
  ...
)
{
  
  # check family
  family <- match.arg(family)
  # convert data.frame to list
  if(is.data.frame(data))
    data <- as.list(data)
  # if(any(sapply(data, is.data.frame)))
  #   stop("Data.frames within the input list are now allowed.")
  # get column names of data
  varnames <- names(data)
  if(is.null(varnames) | any(varnames==""))
    stop("If data is a list, names must be given.")
  # for convenience transform NULL to list(NULL) for list_of_deep_models
  if(missing(list_of_deep_models) | is.null(list_of_deep_models)){ 
    list_of_deep_models <- list(NULL)
    warning("No deep model specified")
  }else if(!is.list(list_of_deep_models)) stop("list_of_deep_models must be a list.")
  # get names of networks
  netnames <- names(list_of_deep_models)
  if(is.null(netnames) & length(list_of_deep_models) > 0)
    netnames <- "d"
  if(!is.null(list_of_deep_models) && is.null(names(list_of_deep_models)))
    names(list_of_deep_models) <- rep("d", length(list_of_deep_models))
  # number of observations
  n_obs <- NROW(y)
  # number of output dim
  output_dim <- NCOL(y)
  # check consistency of #parameters
  nr_params <- length(list_of_formulae)
  if(is.null(dist_fun)) 
    nrparams_dist <- make_tfd_dist(family, return_nrparams = TRUE) else
      nrparams_dist <- nr_params
  if(nrparams_dist < nr_params)
  {
    warning("More formulae specified than parameters available.",
            " Will only use ", nrparams_dist, " formula(e).") 
    nr_params <- nrparams_dist
    list_of_formulae <- list_of_formulae[1:nr_params]
  }
  # check list of formulae is always one-sided
  if(any(sapply(list_of_formulae, function(x) attr( terms(x) , "response" ) != 0 ))){
    stop("Only one-sided formulas are allowed in list_of_formulae.")
  }
  
  # parse formulae
  parsed_formulae_contents <- lapply(list_of_formulae,
                                     get_contents,
                                     data = data,
                                     df = df,
                                     variable_names = varnames,
                                     network_names = netnames,
                                     defaultSmoothing = defaultSmoothing,
                                     null.space.penalty = null.space.penalty)
  
  # check for zero ncol linterms
  for(i in 1:nr_params){
    if(NCOL(parsed_formulae_contents[[i]]$linterms)==0)
      parsed_formulae_contents[[i]]["linterms"] <- list(NULL)
  }

  this_OX <- lapply(parsed_formulae_contents, make_orthog, retcol = TRUE)

  # are parameters trained together?
  if(train_together & !all(sapply(list_of_deep_models, is.null)))
  {

    # check if d-terms and corresponding covariates are in all
    # parameter
    nr_params <- length(parsed_formulae_contents)
    if(length(list_of_deep_models) > 1)
      stop("If train_together=TRUE, a list with", 
           " only one deep learning model should be provided.")
    units_last_layer <- as.list(body(list_of_deep_models[[1]])[[3]])$units
    if(units_last_layer != nr_params)
      stop("The number of units in the last layer of the network ",
           "must be equal to the number of parameters.")
    
  }

  # extract constraints
  # TODO

  # get columns per term
  ncol_deep <- lapply(lapply(
    parsed_formulae_contents, "[[", "deepterms"), function(x){
      ret <- lapply(x, nCOL)
      names(ret) <- names(x)
      return(ret)
    })
      
  ncol_structured <- sapply(
    parsed_formulae_contents[!sapply(parsed_formulae_contents,is.null)],
    function(x){
      ncolsmooth <- 0
      if(!is.null(x[['smoothterms']]))
        ncolsmooth <- sum(sapply(x[['smoothterms']], function(st) if(length(st)>1)
          sum(sapply(st, function(y) NCOL(y$X))) else NCOL(st[[1]]$X)))

      return(ncol_lint(x[['linterms']]) + ncolsmooth)

    })
  # create structured layers
  list_structured <- lapply(1:length(parsed_formulae_contents), function(i)
                            get_layers_from_s(parsed_formulae_contents[[i]], i,
                                              variational = variational,
                                              posterior_fun = posterior_fun,
                                              prior_fun = prior_fun
                                              ))
  
  if(train_together){
    ncol_deep <- ncol_deep[[1]]
    for(i in 2:nr_psarams)
      parsed_formulae_contents[[i]]["deepterms"] <- list(NULL)
  }
    
  # initialize the model
  model <- deepregression_init(
    n_obs = n_obs,
    ncol_structured = ncol_structured,
    ncol_deep = ncol_deep,
    list_structured = list_structured,
    list_deep = list_of_deep_models,
    nr_params = nr_params,
    lss = TRUE,
    train_together = train_together,
    family = family,
    variational = variational,
    dist_fun = dist_fun,
    kl_weight = 1 / n_obs,
    orthogX = this_OX,
    lambda_lasso = lambda_lasso,
    lambda_ridge = lambda_ridge,
    monitor_metric = monitor_metric,
    optimizer = optimizer,
    output_dim = output_dim,
    mixture_dist = mixture_dist,
    split_fun = split_fun,
    posterior = posterior_fun,
    prior = prior_fun,
    ind_fun = ind_fun,
    extend_output_dim = extend_output_dim,
    ...
    )

  # check distribution wrt to specified parameters
  # (not when distfun is given)
  if(is.null(dist_fun)) 
    nrparams_dist <- make_tfd_dist(family, return_nrparams = TRUE) else
      nrparams_dist <- nr_params
  
  # get covariates
  #
  # must be a list of the following form:
  # list(deep_part_param1, deep_part_param2, ..., deep_part_param_u,
  #      deep_struct_param1, deep_struct_param2, ..., deep_struct_param_r)
  parsed_formulae_contents <- lapply(parsed_formulae_contents, orthog_smooth)
  input_cov <- make_cov(parsed_formulae_contents)
  ox <- lapply(parsed_formulae_contents, make_orthog)
  
  input_cov <- unname(c(input_cov, 
                 unlist(lapply(ox[!sapply(ox,is.null)],
                               function(x_per_param) 
                                 unlist(lapply(x_per_param[!sapply(x_per_param,is.null)], 
                                               function(x)
                                   tf$constant(x, dtype="float32")))), 
                        recursive = F)
  ))

  param_names <- names(parsed_formulae_contents)
  l_names_effets <- lapply(parsed_formulae_contents, get_names)
  ind_structterms <- lapply(parsed_formulae_contents, get_indices)

  if(!is.null(validation_data)){
    if(!is.list(validation_data) && length(validation_data)!=2 | 
       is.data.frame(validation_data))
      stop("Validation data must be a list of length two ",
           "with first entry for the data (features) and second entry for response.")
    if(is.data.frame(validation_data[[1]])) 
      validation_data[[1]] <- as.list(validation_data[[1]])
    validation_data[[1]] <- prepare_newdata(parsed_formulae_contents,
                                            validation_data[[1]],
                                            pred = TRUE)
  }
  
  if(!is.null(cv_folds))
  {
    
    validation_split <- NULL
    validation_data <- NULL
    if(!is.list(cv_folds) & is.numeric(cv_folds) & is.null(dim(cv_folds))){
      
      if(cv_folds <= 0) stop("cv_folds must be a positive integer, but is ", 
                             cv_folds, ".")
      cv_folds <- make_cv_list_simple(data_size=NROW(data[[1]]), round(cv_folds), 
                                      seed)
      
    }
  }
    

  ret <- list(model = model,
              init_params =
                list(
                  input_cov = input_cov,
                  n_obs = n_obs,
                  y = y,
                  validation_split = validation_split,
                  validation_data = validation_data,
                  cv_folds = cv_folds,
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
#' @export deepregression_init
#'
deepregression_init <- function(
  n_obs,
  ncol_structured,
  ncol_deep,
  list_structured,
  list_deep,
  use_bias_in_structured = FALSE,
  nr_params = 2,
  lss = TRUE,
  train_together = FALSE,
  lambda_lasso=NULL,
  lambda_ridge=NULL,
  family,
  dist_fun = NULL,
  variational = TRUE,
  weights = NULL,
  learning_rate = 0.01,
  optimizer = optimizer_adam(lr = learning_rate),
  monitor_metric = list(),
  posterior = posterior_mean_field,
  prior = prior_trainable,
  orthog_fun = orthog,
  orthogX = NULL,
  residual_projection = FALSE,
  kl_weight = 1 / n_obs,
  output_dim = 1,
  mixture_dist = FALSE,
  split_fun = split_model,
  ind_fun = function(x) x,
  extend_output_dim = 0
  )
{

  # check injection
  # if(length(inject_after_layer) > nr_params)
  #   stop("Can't have more injections than parameters.")
  # if(any(sapply(inject_after_layer, function(x) x%%1!=0)))
  #   stop("inject_after_layer must be a positive / negative integer")
  
  if(variational){ 
    dense_layer <- function(x, ...)
      layer_dense_variational(x,
        make_posterior_fn = posterior,
        make_prior_fn = prior,
        kl_weight = kl_weight,
        ...
      )
  }else{
    dense_layer <- function(x, ...)
      layer_dense(x, ...)
  }
    
  
  # define the input layers
  inputs_deep <- lapply(ncol_deep, function(param_list){
    lapply(param_list, function(nc){
    if(sum(unlist(nc))==0) return(NULL) else{
      if(is.list(nc) & length(nc)>1){ 
        layer_input(shape = list(sum(unlist(nc))))
      }else if(is.list(nc) & length(nc)==1){
        layer_input(shape = as.list(nc[[1]]))
      }else stop("Not implemented yet.")
    }
    })
  })
  inputs_struct <- lapply(1:length(ncol_structured), function(i){
    nc = ncol_structured[i]
    if(nc==0) return(NULL) else
      # if(!is.null(list_structured[[i]]) & nc > 1)
        # nc>1 will cause problems when implementing ridge/lasso
          layer_input(shape = list(nc))
  })

  if(!is.null(orthogX)){
    ox <- lapply(orthogX, function(x) if(is.null(x)) return(NULL) else{
      lapply(x, function(y){
        if(is.null(y) || y==0) return(NULL) else return(layer_input(shape = list(y)))})
    })
  }
  
  # extend one or more layers' output dimension
  if(length(extend_output_dim) > 1 || extend_output_dim!=0){
    output_dim <- output_dim + extend_output_dim
  }else{
    output_dim <- rep(output_dim, length(inputs_struct))
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
                                   if(!is.null(lambda_lasso) & is.null(lambda_ridge)){
                                     l1 = tf$keras$regularizers$l1(l=lambda_lasso)
                                     return(inputs_struct[[i]] %>%
                                              dense_layer(
                                                units = output_dim[i], 
                                                activation = "linear",
                                                use_bias = use_bias_in_structured,
                                                kernel_regularizer = l1,
                                                name = paste0("structured_lasso_",
                                                              i))
                                     )
                                   }else if(!is.null(lambda_ridge) & is.null(lambda_lasso)){ 
                                     l2 = tf$keras$regularizers$l2(l=lambda_ridge)
                                     return(inputs_struct[[i]] %>%
                                              dense_layer(
                                                units = output_dim[i], 
                                                activation = "linear",
                                                use_bias = use_bias_in_structured,
                                                kernel_regularizer = l2,
                                                name = paste0("structured_ridge_",
                                                              i))
                                     )
                                   }else if(!is.null(lambda_ridge) & !is.null(lambda_lasso)){
                                     l12 = tf$keras$regularizers$l1_l2(l1=lambda_lasso,
                                                                       l2=lambda_ridge)
                                     return(inputs_struct[[i]] %>%
                                              dense_layer(
                                                units = output_dim[i], 
                                                activation = "linear",
                                                use_bias = use_bias_in_structured,
                                                kernel_regularizer = l12,
                                                name = paste0("structured_elastnet_",
                                                              i))
                                     )
                                   }else{
                                     return(inputs_struct[[i]] %>%
                                              dense_layer(
                                                units = output_dim[i], 
                                                activation = "linear",
                                                use_bias = use_bias_in_structured,
                                                name = paste0("structured_linear_",
                                                              i))
                                     )
                                   }
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

  # if(!train_together & 
  #    (length(inputs_deep[!sapply(inputs_deep,is.null)]) != 
  #     length(list_deep[!sapply(list_deep,is.null)])) & 
  #    any(!sapply(inputs_deep, is.null)) & length(ncol_deep)>1)
  #   stop(paste0("If paramters of distribution are not trained together, ",
  #        "a deep model must be provided for each parameter."))
  deep_split <- lapply(ncol_deep, function(param_list){
    lapply(names(param_list), function(nn){
      if(is.null(nn)) return(NULL) else
        split_fun(list_deep[[nn]], -1)
    })
  })

  list_deep <- lapply(deep_split, function(param_list) 
    lapply(param_list, "[[", 1))
  list_deep_ontop <- lapply(deep_split, function(param_list) 
    lapply(param_list, "[[", 2))

  # define deep predictor
  deep_parts <- lapply(1:length(inputs_deep), function(i)
    if(is.null(inputs_deep[[i]]) | length(inputs_deep[[i]])==0) 
      return(NULL) else 
      lapply(1:length(list_deep[[i]]), function(j)
        list_deep[[i]][[j]](inputs_deep[[i]][[j]])))

  ############################################################
  ################# Apply Orthogonalization ##################
  
  # create final linear predictor per distribution parameter
  # -> depending on the presence of a deep or structured part
  # the corresponding part is returned. If both are present
  # the deep part is projected into the orthogonal space of the
  # structured part
  
  # Check if only one shared deep network is present
  if(train_together & !is.null(deep_parts[[1]])){

    if(length(deep_parts) > 1)
      stop("Training deep parts together for more than one deep model",
           " not supported yet.")
    
    if(!all(sapply(ox, is.null))){
      warning("Orthogonalization currently only works with separate deep models.")
      ox <- ox[1] 
    }
    
    # apply orthogonalization
    if(!is.null(ox[[1]]))
      deep_parts[[1]] <- orthog_fun(deep_parts[[1]],
                                    ox[[1]])
    
    # function for split deep model parts
    split_fun <- function(x)
      tf$split(x, num_or_size_splits = nr_params, axis = 1L)

    # apply splitting
    deep_parts <- layer_lambda(list_deep_ontop[[1]](deep_parts[[1]]), 
                               f = split_fun)
    
    list_deep_ontop <- lapply(1:nr_params, function(i) function(obj) obj)

  }
  
  list_pred_param <- lapply(1:nr_params, function(i){
    
    if(length(deep_parts[[i]]) < i) this_deep <- NULL else 
      this_deep <- deep_parts[[i]]
    if(length(list_deep_ontop[[i]]) < i) this_ontop <- NULL else 
      this_ontop <- list_deep_ontop[[i]]
    if(length(structured_parts) < i) this_struct <- NULL else 
      this_struct <- structured_parts[[i]]
    if(length(ox) < i | train_together) this_ox <- NULL else
      this_ox <- ox[[i]]
    
    combine_model_parts(deep = this_deep,
                        deep_top = this_ontop,
                        struct = this_struct,
                        ox = this_ox,
                        orthog_fun = orthog_fun)
  }
  )
  

  # concatenate predictors
  # -> just to split them later again?
  if(length(list_pred_param) > 1)
    preds <- layer_concatenate(list_pred_param) else
      preds <- list_pred_param[[1]]
  
  if(mixture_dist){
    list_pred <- layer_lambda(preds, 
                              f = function(x)
                                {
                                tf$split(x, num_or_size_splits = 
                                           c(1L, as.integer(nr_params-1)),
                                         axis = 1L)
                                })
    list_pred[[1]] <- list_pred[[1]] %>% 
      dense_layer(units = mixture_dist, 
                  activation = "softmax", 
                  use_bias = FALSE)
    preds <- layer_concatenate(list_pred)
  }

  ############################################################
  ### Define Distribution Layer and Variational Inference ####

   
  
  # define the distribution function applied in the last layer

  if(lss){

    # special families needing transformations
    if(family %in% c("betar", "gammar", "negbinom")){
      
      # trafo_list <- family_trafo_funs(family)
      # predsTrafo <- layer_lambda(object = preds, f = trafo_fun)
      # preds <- layer_concatenate(predsTrafo)
      
      dist_fun <- family_trafo_funs_special(family)
      
    }
    
    # apply the transformation for each parameter
    # and put in the right place of the distribution
    if(is.null(dist_fun))
      dist_fun <- make_tfd_dist(family)

    # make model variational and output distribution
    # if(variational){
    # 
    #   out <- preds %>%
    #     layer_dense_variational(
    #       units = length(nr_params),
    #       make_posterior_fn = posterior,
    #       make_prior_fn = prior,
    #       kl_weight = kl_weight
    #     ) %>%
    #     layer_distribution_lambda(dist_fun)
    # 
    # }else{

      out <- preds %>%
        layer_distribution_lambda(dist_fun) 

    # }

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
    unlist(inputs_deep[!sapply(inputs_deep, is.null)],
           recursive = F),
    inputs_struct[!sapply(inputs_struct, is.null)],
    unlist(ox[!sapply(ox, is.null)]))
  )
  # the final model is defined by its inputs
  # and outputs

  model <- keras_model(inputs = inputList,
                       outputs = out)

  # define weights to be equal to 1 if not given
  if(is.null(weights)) weights <- 1
  
  # the negative log-likelihood is given by the negative weighted
  # log probability of the model
  negloglik <- function(y, model) 
    - weights * (model %>% ind_fun() %>% tfd_log_prob(y))

  # compile the model using the defined optimizer,
  # the negative log-likelihood as loss funciton
  # and the defined monitoring metrics as metrics
  model %>% compile(optimizer = optimizer,
                    loss = negloglik,
                    metrics = monitor_metric)

  return(model)

}

