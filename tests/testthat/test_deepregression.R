context("main entry: deepregression")

# helper function to check object dims
expect_object_dims = function(mod, data, loc = c(1,1), scale = c(1,1)) {
  dims = lapply(coef(mod), function(x) dim(x$structured_linear))
  expect_equal(dims[[1]], loc)
  expect_equal(dims[[2]], scale)

  mod %>% fit(epochs=2L, verbose = FALSE, view_metrics = FALSE)

  dims = lapply(coef(mod), function(x) dim(x$structured_linear))
  expect_equal(dims[[1]], loc)
  expect_equal(dims[[2]], scale)

  mean <- mod %>% fitted()
  expect_true(length(mean) == nrow(data))
  expect_true(is.numeric(mean))

  mean <- mod %>% sd(data)
  expect_true(length(mean) == nrow(data))
  expect_true(is.numeric(mean))
}



test_that("simple additive model", {

  n <- 1500
  deep_model <- function(x) x %>%
    layer_dense(units = 2L, activation = "relu", use_bias = FALSE) %>%
    layer_dense(units = 1L, activation = "linear")

  x <- runif(n) %>% as.matrix()
  true_mean_fun <- function(xx) sin(10 * apply(xx, 1, mean) + 1)

  for (i in c(1, 3, 50)) {
    data = data.frame(matrix(x, ncol=i))
    if (ncol(data) == 1L) colnames(data) = "X1"
    y <- true_mean_fun(data)
    mod <- deepregression(
      y = y,
      data = data,
      list_of_formulae = list(loc = ~ 1 + d(X1), scale = ~1),
      list_of_deep_models = list(d = deep_model)
    )
    expect_object_dims(mod, data)
  }

  # 2 deep 1 structured + intercept
  data = data.frame(matrix(x, ncol=3))
  y <- true_mean_fun(data)
  mod <- deepregression(
    y = y,
    data = data,
    list_of_formulae = list(loc = ~ X3 + d(X1) + g(X2), scale = ~1),
    list_of_deep_models = list(d = deep_model, g = deep_model)
  )
  expect_object_dims(mod, data, c(2, 1))


  # 2 deep 1 structured no intercept
  mod <- deepregression(
    y = y,
    data = data,
    list_of_formulae = list(loc = ~ -1 + X3 + d(X1) + g(X2), scale = ~1),
    list_of_deep_models = list(d = deep_model, g = deep_model)
  )
  expect_object_dims(mod, data, c(1, 1))
})
