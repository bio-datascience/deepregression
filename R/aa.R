.onLoad <- function(libname, pkgname) {
  reticulate::configure_environment(pkgname)

  if(!reticulate::py_module_available("tensorflow"))
  {
    
    if(!py_module_available(six)) reticulate::py_install("six")
    
    try({
      tensorflow::install_tensorflow(version = "2.0.0")
      tfprobability::install_tfprobability(version = "0.8.0", tensorflow = "2.0.0")
      keras::install_keras(tensorflow = "2.0.0")
    })
    
  }
  # Use TF
  suppressMessages(try(keras::use_implementation("tensorflow"), silent = TRUE))
  # catch TFP error
  suppressMessages(try(david <- tfprobability::tfd_normal(0,1), silent = TRUE))
}