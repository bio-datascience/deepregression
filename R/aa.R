.onLoad <- function(libname, pkgname) {
  reticulate::configure_environment(pkgname)
  suppressMessages(try(david <- tfd_normal(0,1), silent = TRUE))
}

reticulate::use_python(python = "/usr/bin/python3",
                       required = TRUE)

if(length(reticulate::virtualenv_list())==0)
  reticulate::virtualenv_create()

if(!reticulate::py_module_available("tensorflow"))
{
  
  library(tensorflow)
  install_tensorflow(version = "2.0.0")
  library(tfprobability)
  install_tfprobability(version = "0.8.0", tensorflow = "2.0.0")
  suppressMessages(try(david <- tfd_normal(0,1), silent = TRUE))
  
}

# install_keras(tensorflow = "2.0")


# keras::use_implementation("tensorflow")
# Sys.setenv(TF_KERAS=1)

# initialize v2 behavior -> make TF 2 required
# tf$compat$v1$enable_v2_behavior()

# catch weird initial loading error
# suppressMessages(try(david <- tfd_normal(0,1), silent = TRUE))
