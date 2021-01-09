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
                           target_size, color_mode, is_trafo = FALSE,
                           x_col, shuffle = TRUE, seed = 42L)
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
  
  if(is.logical(data_tab[[1]][[1]]) || is.na(data_tab[[1]][[1]][[1]]))
      data_tab[[1]] <- data_tab[[1]][-1]
  
  ldt <- length(data_tab[[1]])
  
  if(ldt>2)
  {
    
    lens <- rep(2, ceiling((ldt-2)/2))
    lens[length(lens)] <- ifelse(ldt%%2==0, 2, 1)
    
    for(i in 1:ceiling((ldt-2)/2)){
      
      this_ind <- (i-1)*2 + 1:lens[i]
      
      if(length(this_ind)>1){
        
        gen_images <- combine_generators_list_unlist(
          gen_images, 
          make_generator_from_matrix(
            x = data_tab[[1]][this_ind], y = NULL, 
            batch_size = batch_size, shuffle = shuffle, seed = seed
          ) 
        )
        
      }else{
      
          gen_images <- combine_generators_unlist_list(
            gen_images, 
            make_generator_from_matrix(
              x = data_tab[[1]][[this_ind]], y = NULL, 
              batch_size = batch_size, shuffle = shuffle, seed = seed
            ) 
          ) 
      }
      
    }
    
    # str(gen_images$`__getitem__`(1L))
    
  }
  
  if(length(data_tab)==1) this_y <- NULL else this_y <- data_tab[[2]]
  gen_tab <- make_generator_from_matrix(
    x = data_tab[[1]][(ldt-1):ldt], y = this_y, 
    batch_size = batch_size, shuffle = shuffle, seed = seed
  ) 
    
  # str(gen_tab$`__next__`())

  if(is.null(this_y)){
    
    if(ldt>2)
      gen <- combine_generators_twolists(gen_images, gen_tab) else
        gen <- combine_generators_list_yless(gen_images, gen_tab)
    
  }else{
    
    if(ldt>2){
      
      gen <- combine_generators_xy( 
        gen_images, gen_tab
      )
      
    }else{
      
      gen <- combine_generators_x( 
        gen_images, gen_tab
      )
      
    }
    
  }
    
  # str(gen$`__getitem__`(1L))

  return(gen)
  
}


# from mlr3keras

#' Make a DataGenerator from a data.frame or matrix
#'
#' Creates a Python Class that internally iterates over the data.
#' @param x matrix;
#' @param y vector;
#' @param generator generator as e.g. obtained from `keras::image_data_generator`.
#'   Used for consistent train-test splits.
#' @param batch_size integer 
#' @param shuffle logical; Should data be shuffled?
#' @param seed integer; seed for shuffling data.
#' @export
make_generator_from_matrix = function(x, y = NULL, generator=image_data_generator(), 
                                      batch_size=32L, shuffle=TRUE, seed=1L) {
  python_path <- system.file("python", package = "deepregression")
  generators <- reticulate::import_from_path("generators", path = python_path)
  generators$Numpy2DArrayIterator(x, y, generator, batch_size=as.integer(batch_size), 
                                  shuffle=shuffle,seed=as.integer(seed))
}


combine_generators = function(gen1, gen2) {
  python_path <- system.file("python", package = "deepregression")
  generators <- reticulate::import_from_path("generators", path = python_path)
  generators$CombinedGenerator(gen1, gen2)
}

combine_generators_x = function(gen1, gen2) {
  python_path <- system.file("python", package = "deepregression")
  generators <- reticulate::import_from_path("generators", path = python_path)
  generators$CombinedGeneratorX(gen1, gen2)
}

combine_generators_xy = function(gen1, gen2) {
  python_path <- system.file("python", package = "deepregression")
  generators <- reticulate::import_from_path("generators", path = python_path)
  generators$CombinedGeneratorXY(gen1, gen2)
}

combine_generators_list = function(gen1, gen2) {
  python_path <- system.file("python", package = "deepregression")
  generators <- reticulate::import_from_path("generators", path = python_path)
  generators$CombinedGeneratorList(gen1, gen2)
}

combine_generators_list_unlist = function(gen1, gen2) {
  python_path <- system.file("python", package = "deepregression")
  generators <- reticulate::import_from_path("generators", path = python_path)
  generators$CombinedGeneratorListUnlist(gen1, gen2)
}

combine_generators_unlist_list = function(gen1, gen2) {
  python_path <- system.file("python", package = "deepregression")
  generators <- reticulate::import_from_path("generators", path = python_path)
  generators$CombinedGeneratorUnlistList(gen1, gen2)
}

combine_generators_twolists = function(gen1, gen2) {
  python_path <- system.file("python", package = "deepregression")
  generators <- reticulate::import_from_path("generators", path = python_path)
  generators$CombinedGeneratorTwoLists(gen1, gen2)
}

combine_generators_list_yless = function(gen1, gen2) {
  python_path <- system.file("python", package = "deepregression")
  generators <- reticulate::import_from_path("generators", path = python_path)
  generators$CombinedGeneratorListYless(gen1, gen2)
}
