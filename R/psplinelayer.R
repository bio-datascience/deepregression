splineLayer <- R6::R6Class("splineLayer",

                           lock_objects = FALSE,
                           inherit = KerasLayer,

                           public = list(

                             output_dim = NULL,

                             kernel = NULL,

                             initialize = function(
                               # input_shape=NULL,
                               shared_weights=FALSE,
                               kernel_regularizer=NULL,
                               use_bias=FALSE,
                               kernel_initializer='glorot_uniform',
                               bias_initializer='zeros'#,
                               # ...
                             ) {

                               # super$initialize(...)

                               # self$input_shape = input_shape
                               self$shared_weights = shared_weights
                               self$kernel_regularizer = tf$keras$regularizers$get(
                                 kernel_regularizer)
                               self$use_bias = use_bias
                               self$kernel_initializer = tf$keras$initializers$get(
                                 kernel_initializer)
                               self$bias_initializer = tf$keras$initializers$get(
                                 bias_initializer)
                               self$input_spec = tf$keras$layers$InputSpec(min_ndim=3)
                             },

                             build = function(input_shape) {

                               stopifnot(length(input_shape) >= 3)

                               n_bases = input_shape[[length(input_shape)]]
                               n_features = input_shape[[length(input_shape)-1]]

                               self$inp_shape = input_shape
                               self$n_features = n_features
                               self$n_bases = n_bases

                               if(self$shared_weights)
                                 use_n_features = 1 else
                                   use_n_features = self$n_features

                               self$kernel =
                                 self$add_weight(shape=list(n_bases, use_n_features),
                                                 initializer=self$kernel_initializer,
                                                 name='kernel',
                                                 regularizer=self$kernel_regularizer,
                                                 trainable=TRUE)

                               if(self$use_bias)
                                 self$bias =
                                 self$add_weight(shape = (n_features),
                                                 # is that the correct pendant to
                                                 # python's (n_features, )?
                                                 initializer=self$bias_initializer,
                                                 name='bias',
                                                 regularizer=NULL)

                               self$built = TRUE

                               super$build(self
                                           #,input_shape
                               )

                             },

                             call = function(inputs, mask = NULL) {
                               N = length(self$inp_shape)

                               if (self$shared_weights)
                                 return(k_squeeze(k_dot(inputs, self$kernel), -1))

                               # print("output def")
                               output =
                                 k_permute_dimensions(inputs, list(
                                   N - 1,
                                   as.list(1:(N-2)), # permute is 1-based
                                   N
                                 )
                                 )

                               # print("output_reshaped def")

                               sh = list(self$n_features, -1, self$n_bases)
                               output_reshaped =
                                 k_reshape(output, sh)

                               # print(str(output_reshaped,1))

                               axes = NULL
                               # # added due to missing translation
                               # # of the argument axes in k_batch_dot
                               # # -_____-
                               # tKernel = k_transpose(self$kernel)
                               # x_shape = k_int_shape(output_reshaped)
                               # y_shape = k_int_shape(tKernel)
                               # x_ndim = length(x_shape)
                               # y_ndim = length(y_shape)
                               # if(y_ndim == 2)
                               #   axes = list(x_ndim - 1, y_ndim - 1) else
                               #     axes = list(x_ndim - 1, y_ndim - 2)
                               # #####################################

                               # print("bd_output def")

                               bd_output =
                                 k_batch_dot(output_reshaped,
                                             k_transpose(self$kernel),
                                             axes = axes)

                               # this one is tricky to translate
                               # from python. Not sure if this is
                               # correct
                               sh = list(self$n_features, -1)
                               if(length(self$inp_shape) > 3){
                                 sh = append(sh, self$inp_shape[2:(N - 2)])
                               }

                               # print("output def 1")

                               output = k_reshape(bd_output, sh)
                               # move axis 0 (features) to back

                               # print("output def 2")

                               # print(str(output,1))


                               output =
                                 k_permute_dimensions(output,
                                                      c(as.list(2:(N-1)), list(1)))

                               # print("done")

                               # permute is 1-based
                               if(self$use_bias)
                                 output = k_bias_add(output,
                                                     self$bias, data_format="channels_last")
                               return(output)
                             },

                             compute_output_shape = function(input_shape) {
                               input_shape[-length(input_shape)]
                             },

                             get_config = function()
                             {
                               config = list(
                                 'shared_weights' = self$shared_weights,
                                 'kernel_regularizer' = tf$keras$regularizers$serialize(
                                   self$kernel_regularizer),
                                 'use_bias' = self$use_bias,
                                 'kernel_initializer' =
                                   tf$keras$initializers$serialize(self$kernel_initializer),
                                 'bias_initializer' =
                                   tf$keras$initializers$serialize(self$bias_initializer)
                               )
                               base_config = super$get_config()
                               # is this the correct analogous way to
                               # python's combining dicts?
                               return(c(base_config, config))
                             }

                           )
)

### instantiate spline layer


#' Wrapper function to create spline layer
#'
#' @param Model or layer object.
#' @param name An optional name string for the layer (should be unique).
#' @param input_shape Input dimensionality not including samples axis.
#' @param lambdas vector of smoothing parameters
#' @param Ps list of penalty matrices
layer_spline <- function(object,
                         name = NULL,
                         trainable = TRUE,
                         input_shape,
                         shared_weights=FALSE,
                         lambdas,
                         Ps,
                         use_bias = FALSE,
                         kernel_initializer = 'glorot_uniform',
                         bias_initializer = 'zeros',
                         ...) {

  bigP = bdiag(lapply(1:length(Ps), function(j) lambdas[j] * Ps[[j]]))

  if(sum(lambdas)==0)
    regul <- NULL else
      regul <- function(x)
        k_mean(k_batch_dot(x, k_dot(
          # tf$constant(
          sparse_mat_to_tensor(bigP),
          # dtype = "float32"),
          x),
          axes=2) # 1-based
        )

    create_layer(layer_class = splineLayer,
                 object = object,
                 args = c(list(input_shape = input_shape),
                          name = name,
                          trainable = trainable,
                          shared_weights=shared_weights,
                          kernel_regularizer=regul,
                          use_bias=use_bias,
                          kernel_initializer=kernel_initializer,
                          bias_initializer=bias_initializer,
                          list(...))
    )
}


#### get layer based on smoothCon object
get_layers_from_s <- function(this_param, nr=NULL)
{

  if(is.null(this_param)) return(NULL)

  lambdas <- c()
  Ps = list()
  params = 0

  # create vectors of lambdas and list of penalties
  if(!is.null(this_param$linterms)){
    lambdas = c(lambdas, rep(0, ncol(this_param$linterms)))
    Ps = c(Ps, list(0)[rep(1, ncol(this_param$linterms))])
    params = ncol(this_param$linterms)
  }
  if(!is.null(this_param$smoothterms)){
    these_lambdas = unlist(sapply(this_param$smoothterms, "[[", "sp"))
    lambdas = c(lambdas, these_lambdas)
    these_Ps = sapply(this_param$smoothterms, "[[", "S")
    is_TP <- sapply(these_Ps, length) > 1
    if(any(is_TP))
      these_Ps[which(is_TP)] <- lapply(these_Ps[which(is_TP)],
                                       function(x){

                                         return(
                                           # isotropic smoothing
                                           # TODO: Check if correct
                                           x[[1]] + x[[2]]
                                           # kronecker(x[[1]],
                                           #           diag(ncol(x[[2]]))) +
                                           #   kronecker(diag(ncol(x[[1]])), x[[2]])
                                         )
                                       })
    s_as_list <- sapply(these_Ps, class)=="list"
    if(any(s_as_list)){
      for(i in which(s_as_list)){
        if(length(these_Ps[i])> 1)
          stop("Can not deal with penalty of smooth term ", names(these_Ps)[i])
        these_Ps[i] <- these_Ps[i][[1]]
      }
    }
    Ps = c(Ps, these_Ps)
    params = params + sum(sapply(these_Ps, NCOL))
  }else{
    # only linear terms
    # name <- "linear"
    # if(!is.null(nr)) name <- paste(name, nr, sep="_")
    return(
      # layer_dense(input_shape = list(params),
      #             units = 1,
      #             use_bias = FALSE,
      #             name = name)
      NULL
    )
  }

  name <- "structured_nonlinear"
  if(!is.null(nr)) name <- paste(name, nr, sep="_")

  layer_spline(input_shape = list(1,params),
               # the one is just an artifact from concise
               name = name,
               lambdas = lambdas,
               Ps = Ps)

}
