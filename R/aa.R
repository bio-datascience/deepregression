.onLoad <- function(libname, pkgname) {
  reticulate::configure_environment(pkgname)

  if(!reticulate::py_available() | 
     !reticulate::py_module_available("tensorflow"))
    check_and_install()
  if(!py_module_available(six)) reticulate::py_install("six")
  # Use TF
  suppressMessages(try(keras::use_implementation("tensorflow"), silent = TRUE))
  # catch TFP error
  suppressMessages(try(david <- tfprobability::tfd_normal(0,1), silent = TRUE))
}