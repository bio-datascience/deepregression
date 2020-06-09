#' Data pipeline for SDDR models
#' 
#' 
#' @param cv_folds a list of lists, each list element has two elements, one for
#' training indices and one for testing indices; 
#' if a single integer number is given, 
#' a simple k-fold cross-validation is defined, where k is the supplied number.
#' @param validation_data data for validation during training.
#' @param validation_split percentage of training data used for validation. 
#' Per default 0.2.
#' 
#' @export
data_pipeline <- function(
  cv_folds = NULL,
  validation_data = NULL,
  validation_split = ifelse(is.null(validation_data) & is.null(cv_folds), 0.2, 0),
)
{
  
  # orthogonalization for smooths
  parsed_formulae_contents <- lapply(parsed_formulae_contents, orthog_smooth,  
                                     zero_cons = zero_constraint_for_smooths)
  
  
  cat("Translating data into tensors...")
  input_cov <- make_cov(parsed_formulae_contents)
  cat(" Done.\n")
  
  # orthogonalize
  ox <- lapply(parsed_formulae_contents, make_orthog)
  
  # define input covariates
  input_cov <- unname(c(input_cov, 
                        unlist(lapply(ox[!sapply(ox,is.null)],
                                      function(x_per_param) 
                                        unlist(lapply(x_per_param[!sapply(x_per_param,is.null)], 
                                                      function(x)
                                                        tf$constant(x, dtype="float32")))), 
                               recursive = F)
  ))
  
  if(!is.null(offset)){
    
    cat("Using an offset.")
    input_cov <- c(input_cov, unlist(lapply(offset[!sapply(offset, is.null)],
                                            function(x) tf$constant(matrix(x, ncol = 1), 
                                                                    dtype="float32")),
                                     recursive = FALSE))
    
  }
  
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
    
    if(!is.null(offset_val)){
      # print("Using an offset.")
      validation_data[[1]] <- c(validation_data[[1]], 
                                unlist(lapply(offset_val[!sapply(offset_val, is.null)],
                                              function(x) tf$constant(matrix(x, ncol = 1), 
                                                                      dtype="float32")),
                                       recursive = FALSE))
    }
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
  
}