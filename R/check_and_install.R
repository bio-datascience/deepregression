#' Function to check python environment and install necessary packages
#' 
#' @return Function that checks if a Python environment is available
#' and contains TensorFlow. If not the recommended version is installed.
#' 
#' @export
#' 
check_and_install <- function()
{
  
  if(!reticulate::py_available()){
    
    if(length(list.files(reticulate::miniconda_path()))==0)
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