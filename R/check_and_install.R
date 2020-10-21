#' Function to check python environment and install necessary packages
#'
#' Note: The package currently relies on tensorflow version 2.0.0 which is
#' not available for the latest python versions 3.9 and later.
#' If you encounter problems with installing the required python modules
#' please make sure, that a correct python version is configured using
#' `py_discover_config` and change the python version if required.
#' Internally uses keras::install_keras.
#'
#' @param force if TRUE, forces the installations
#' @return Function that checks if a Python environment is available
#' and contains TensorFlow. If not the recommended version is installed.
#'
#' @export
check_and_install <- function(force = FALSE) {
  if (!reticulate::py_module_available("tensorflow") || force) {
    keras::install_keras(tensorflow = "2.0", extra_packages = c("tfprobability==0.8", "six"))
  } else {
    message("Tensorflow found, skipping tensorflow installation!")
    if (!reticulate::py_module_available("tfprobability") || !reticulate::py_module_available("six")) {
      message("Installing pytho modules 'tfprobability' and 'six'")
      reticulate::py_install(packages = c("tfprobability==0.8", "six"))
    }
  }
}
