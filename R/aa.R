.onLoad <- function(libname, pkgname) {
  reticulate::configure_environment(pkgname)
  
  # check if TF is available
  # if not, 
  if(!reticulate::py_available())
  {
    
    # first check if an env is available
    if(!reticulate::py_available()){
      
      if(length(list.files(reticulate:::miniconda_path()))==0)
        reticulate::install_miniconda()
      reticulate::use_miniconda(required = TRUE)
      
    }
    
    if(!reticulate::py_module_available("tensorflow"))
    {
      
      tensorflow::install_tensorflow(version = "2.0.0")
      
      # if(!reticulate::py_module_available("tfprobability"))
      tfprobability::install_tfprobability(version = "0.8.0", tensorflow = "2.0.0")
      
    }
  }
  
  # Use TF
  keras::use_implementation("tensorflow")
  
  # catch TFP error
  suppressMessages(try(david <- tfd_normal(0,1), silent = TRUE))
}