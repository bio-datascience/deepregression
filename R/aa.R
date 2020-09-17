.onLoad <- function(libname, pkgname) {
  # reticulate::configure_environment(pkgname)

  # Use TF
  suppressMessages(try(keras::use_implementation("tensorflow"), silent = TRUE))
  # catch TFP error
  suppressMessages(try(david <- tfprobability::tfd_normal(0,1), silent = TRUE))
}