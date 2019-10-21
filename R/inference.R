prior_trainable <-
  function(kernel_size,
           bias_size = 0,
           dtype = NULL) {
    n <- kernel_size + bias_size
    keras_model_sequential() %>%
      # we'll comment on this soon
      # layer_variable(n, dtype = dtype, trainable = FALSE) %>%
      layer_variable(n, dtype = dtype, trainable = TRUE) %>%
      layer_distribution_lambda(function(t) {
        tfd_independent(tfd_normal(loc = t, scale = 1),
                        reinterpreted_batch_ndims = 1)
      })
  }


posterior_mean_field <-
  function(kernel_size,
           bias_size = 0,
           dtype = NULL) {
    n <- kernel_size + bias_size
    c <- log(expm1(1))
    keras_model_sequential(list(
      layer_variable(shape = 2 * n, dtype = dtype),
      layer_distribution_lambda(
        make_distribution_fn = function(t) {
          tfd_independent(tfd_normal(
            loc = t[1:n],
            scale = 1e-5 + tf$nn$softplus(c + t[(n + 1):(2 * n)])
          ), reinterpreted_batch_ndims = 1)
        }
      )
    ))
  }