#' Generator for placeholder object
#'
#' @param var variable to be transformed; each entry must be a path to an
#' existing file
#' @param read_method function that is used to load the object from the path
#' @param dim the dimension of the actual data; if NULL, \code{read_method} is
#' applied to the first entry of \code{var}.
#'
#' @export
#'
path_to_placeholder <- function(var, read_method, dim = NULL)
{
  
  # check if data exists
  if(!all(sapply(var, file.exists)))
    stop("Not all files exist in the path variable. Can't create placeholder.")
  
  # check dimension
  if(is.null(dim))
    dim <- dim(read_method(var[1]))
  
  # create placeholder
  class(var) <- c("placeholder", "character")
  attr(var, "dims") <- c(length(var),dim)
  attr(var, "read_method") <- read_method
  
  return(var)
  
}

#' Methods for placeholder class
#' 
#' @method dim placeholder
#' @export
#' @rdname methodPlaceholder
dim.placeholder <- function(x) attr(x, "dims")

#' @export
#' @rdname methodPlaceholder
`[.placeholder` <- function(x, i, ...) {
  path_to_placeholder(NextMethod(),  read_method = attr(x, "read_method"), dim = attr(x, "dim"))
}

#' creates a generator for training
#'
#' @param batch_size integer
#' @param max_data maximum number of samples
#' @param phs indicator for placeholders
#' @param input_x x input
#' @param input_y y input
#'  

make_generator <- function(batch_size, max_data, phs, x, y)
{
  
  id <- 1
  
  return(
    generator <- function()
    {
      
      ind <- (id-1)*batch_size + 1:batch_size
      ind[2] <- pmin(ind[2], max_data)
      input_x_batch <- subset_input_cov(x, ind)
      input_x_batch[phs] <- 
        lapply(input_x_batch[phs], function(xx) attr(xx, "read_method")(xx))
      id <<- id+1
      if((id-1)*batch_size+1 > max_data) id <<- 1
      return(list(input_x_batch, subset_array(y, ind)))
    }
  )
      
}
