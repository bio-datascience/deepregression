diffuse_scale <- 1000

prior_trainable <-
  function(kernel_size,
           bias_size = 0,
           dtype = NULL) {
    n <- kernel_size + bias_size
    keras_model_sequential() %>%
      layer_variable(n, dtype = dtype, trainable = TRUE) %>%
      layer_distribution_lambda(function(t) {
        tfd_independent(tfd_normal(loc = t, scale = diffuse_scale),
                        reinterpreted_batch_ndims = 1)
      })
  }


posterior_mean_field <-
  function(kernel_size,
           bias_size = 0,
           dtype = NULL) {
    n <- kernel_size + bias_size
    keras_model_sequential(list(
      layer_variable(shape = 2 * n, dtype = dtype),
      layer_distribution_lambda(
        make_distribution_fn = function(t) {
          tfd_independent(tfd_normal(
            loc = t[1:n],
            scale = 1e-8 + tf$nn$softplus(log(expm1(1)) + t[(n + 1):(2 * n)])
          ), reinterpreted_batch_ndims = 1)
        }
      )
    ))
  }

prior_pspline <- 
  function(kernel_size,
           bias_size = 0,
           dtype = 'float32',
           P) {
    n <- kernel_size + bias_size
    keras_model_sequential() %>% 
      layer_variable(n, dtype = dtype, trainable = TRUE) %>% 
      layer_distribution_lambda(function(t) {
        tfd_multivariate_normal_full_covariance(
          loc = t,#tf$constant(rep(0, length(t)), dtype="float32"), 
          covariance_matrix = tf$constant(P, dtype="float32")
          )
      })
    
  }
