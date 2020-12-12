#' 
#' #' Generator for placeholder object
#' #'
#' #' @param var variable to be transformed; each entry must be a path to an
#' #' existing file
#' #' @param read_method function that is used to load the object from the path
#' #' @param dim the dimension of the actual data; if NULL, \code{read_method} is
#' #' applied to the first entry of \code{var}.
#' #'
#' #' @export
#' #'
#' path_to_placeholder <- function(var, read_method, dim = NULL)
#' {
#'   
#'   # check if data exists
#'   if(!all(sapply(var, file.exists)))
#'     stop("Not all files exist in the path variable. Can't create placeholder.")
#'   
#'   # check dimension
#'   if(is.null(dim))
#'     dim <- dim(read_method(var[1]))
#'   
#'   # create placeholder
#'   class(var) <- c("placeholder", "character")
#'   attr(var, "dims") <- c(length(var),dim)
#'   attr(var, "read_method") <- read_method
#'   
#'   return(var)
#'   
#' }
#' 
#' #' Methods for placeholder class
#' #' 
#' #' @method dim placeholder
#' #' @export
#' #' @rdname methodPlaceholder
#' dim.placeholder <- function(x) attr(x, "dims")
#' 
#' #' @export
#' #' @rdname methodPlaceholder
#' `[.placeholder` <- function(x, i, ...) {
#'   path_to_placeholder(NextMethod(),  read_method = attr(x, "read_method"), dim = attr(x, "dim"))
#' }
#' 

#' creates a generator for training
#'
#' @param data
#' @param batch_size integer
#' @param x_col name of image column
#' @param shuffle logical for shuffling data
#' @param seed
#'  
make_generator <- function(data_image, data_tab, batch_size, 
                           target_size, color_mode,
                           x_col, shuffle = TRUE, seed = 42L,
                           generator)
{
  
  gen_images <- flow_images_from_dataframe(data_image, 
                                           x_col = x_col, 
                                           class_mode = NULL,
                                           target_size = target_size,
                                           color_mode = color_mode,
                                           batch_size = batch_size, 
                                           shuffle = shuffle, 
                                           seed = seed)
  
  # str(gen_images$`__next__`())
  
  gen_tab <- BatchDataIteratorX(data_tab[[1]], 
                                batch_size = batch_size, 
                                shuffle = shuffle, 
                                seed = seed)
  # )
  
  # str(gen_tab$`__next__`())
  
  gen_y <- BatchDataIterator(data_tab[[2]], 
                             batch_size = batch_size, 
                             shuffle = shuffle, 
                             seed = seed
  )
  
  # str(gen_y$`__next__`())
  
  gen <- MultiIteratorImageTab(list(gen_images, gen_tab))
  
  # str(gen$`__next__`())
  
  gen_all <- MultiIteratorXY(c(gen, gen_y))
  
  # str(gen_all$`__next__`())
  
  return(gen_all)
      
}


# Parallel Iterator over N data generators
MultiIteratorXY = reticulate::PyClass("MultiIteratorXY",
                                      defs = list(
                                        seqs = NULL,
                                        `__init__` = function(self, seqs, shuffle = TRUE){
                                          self$seqs = seqs
                                          super()$`__init__`(
                                            n = as.integer(seqs[[1]]$n),
                                            batch_size = as.integer(seqs[[1]]$batch_size),
                                            shuffle = shuffle,
                                            seed = as.integer(self$seqs[[1]]$seed)
                                          )
                                          return(NULL)
                                        },
                                        `__len__` = function(self){
                                          self$seqs[[1]]$`__len__`()
                                        },
                                        `_get_batches_of_transformed_samples` = function(self, index_array) {
                                          xys = purrr::map(self$seqs, function(x) {
                                            x$`_get_batches_of_transformed_samples`(as.integer(index_array))
                                          })
                                          #list(purrr::map(xys, 1L), xys[[1]][[2]])
                                        }
                                        # `__getitem__` = function(self, idx) {
                                        #   xys = map(self$seqs, function(x) {
                                        #     x$`__getitem__`(idx)[[1]]
                                        #   })
                                        #  list(map(xys, 1L), xys[[1]][[2]])
                                        # },
                                        # `__next__` = function(self, idx) {
                                        #   xys = map(self$seqs, function(x) {
                                        #     x$`__getitem__`(idx)[[1]]
                                        #   })
                                        #   list(map(xys, 1L), xys[[1]][[2]])
                                        # }
                                      ),
                                      inherit = tensorflow::tf$keras$preprocessing$image$Iterator
)

MultiIteratorImageTab = reticulate::PyClass("MultiIteratorImageTab",
                                            defs = list(
                                              seqs = NULL,
                                              `__init__` = function(self, seqs, shuffle = TRUE){
                                                self$seqs = seqs
                                                super()$`__init__`(
                                                  n = as.integer(seqs[[1]]$n),
                                                  batch_size = as.integer(seqs[[1]]$batch_size),
                                                  shuffle = shuffle,
                                                  seed = as.integer(self$seqs[[1]]$seed)
                                                )
                                                return(NULL)
                                              },
                                              `__len__` = function(self){
                                                self$seqs[[1]]$`__len__`()
                                              },
                                              `_get_batches_of_transformed_samples` = function(self, index_array) {
                                                c(list(
                                                  self$seqs[[1]]$`_get_batches_of_transformed_samples`(
                                                    as.integer(index_array))
                                                  ),
                                                  self$seqs[[2]]$`_get_batches_of_transformed_samples`(
                                                    as.integer(index_array))
                                                )
                                              }
                                              # `__getitem__` = function(self, idx) {
                                              #   xys = map(self$seqs, function(x) {
                                              #     x$`__getitem__`(idx)[[1]]
                                              #   })
                                              #  list(map(xys, 1L), xys[[1]][[2]])
                                              # },
                                              # `__next__` = function(self, idx) {
                                              #   xys = map(self$seqs, function(x) {
                                              #     x$`__getitem__`(idx)[[1]]
                                              #   })
                                              #   list(map(xys, 1L), xys[[1]][[2]])
                                              # }
                                            ),
                                            inherit = tensorflow::tf$keras$preprocessing$image$Iterator
)


## Iterator over a list of tensors
BatchDataIterator = reticulate::PyClass("BatchDataIterator",
                                        defs = list(
                                          data = NULL,
                                          `__init__` = function(self, data, batch_size=32L, shuffle=TRUE, seed = 1L){
                                            self$data = data
                                            n = as.integer(nrow(data))
                                            super()$`__init__`(
                                               n = n,
                                               batch_size = as.integer(batch_size),
                                               shuffle = shuffle,
                                               seed = as.integer(seed)
                                             )
                                             return(NULL)
                                           },
                                           `_get_batches_of_transformed_samples` = function(self, index_array){
                                             
                                             self$data[index_array, ,drop=FALSE]
                                             
                                           }
                                         ),
                                         inherit = tensorflow::tf$keras$preprocessing$image$Iterator
)

## Iterator over a list of tensors
BatchDataIteratorX = reticulate::PyClass("BatchDataIteratorX",
                                         defs = list(
                                           data = NULL,
                                           `__init__` = function(self, data, batch_size=32L, shuffle=TRUE, seed = 1L){
                                             self$data = data
                                             n = as.integer(nrow(data[[1]]))
                                             super()$`__init__`(
                                               n = n,
                                               batch_size = as.integer(batch_size),
                                               shuffle = shuffle,
                                               seed = as.integer(seed)
                                             )
                                             return(NULL)
                                           },
                                           `_get_batches_of_transformed_samples` = function(self, index_array){
                                             
                                             lapply(self$data, function(x) x[index_array, ,drop=FALSE])
                                             
                                           }
                                         ),
                                         inherit = tensorflow::tf$keras$preprocessing$image$Iterator
)

# iterate over vector / matrix
BatchDataIteratorY = reticulate::PyClass("BatchDataIteratorY",
                                         defs = list(
                                           data = NULL,
                                           `__init__` = function(self, data, batch_size=32L, shuffle=TRUE, seed = 1L){
                                             self$data = data
                                             n = as.integer(NROW(data))
                                             super()$`__init__`(
                                               n = n,
                                               batch_size = as.integer(batch_size),
                                               shuffle = shuffle,
                                               seed = as.integer(seed)
                                             )
                                             return(NULL)
                                           },
                                           `_get_batches_of_transformed_samples` = function(self, index_array){
                                             
                                             if(is.null(dim(data))) self$data[index_array] else
                                               self$data[index_array,]
                                             
                                           }
                                         ),
                                         inherit = tensorflow::tf$keras$preprocessing$image$Iterator
)

