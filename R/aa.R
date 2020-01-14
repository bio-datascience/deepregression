.onLoad <- function(libname, pkgname) {
  reticulate::configure_environment(pkgname)
}

# initialize v2 behavior -> make TF 2 required
tf$compat$v1$enable_v2_behavior()
