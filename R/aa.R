.onLoad <- function(libname, pkgname) {
  reticulate::configure_environment(pkgname)
  
  # check if TF is available
  is_TF_avail <- require(tensorflow)
  if(class(is_TF_avail)=="try-error")
    is_TF_avail <- FALSE
  
  # if not, 
  if(!is_TF_avail)
  {
    
    # first check if an env is available
    if(!reticulate::py_available()){
      
      reticulate::install_miniconda()
      reticulate::use_miniconda(required = TRUE)
      
    }
    
    library(tensorflow)
    install_tensorflow(version = "2.0.0")
    library(tfprobability)
    install_tfprobability(version = "0.8.0", tensorflow = "2.0.0")
    suppressMessages(try(david <- tfd_normal(0,1), silent = TRUE))
    
  }
  
  # Use TF
  keras::use_implementation("tensorflow")
  
  # catch TFP error
  suppressMessages(try(david <- tfd_normal(0,1), silent = TRUE))
}