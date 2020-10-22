context("various")

test_that("check_and_install", {
  expect_message(check_and_install(), "Tensorflow found, skipping tensorflow installation!")
})


test_that("callbacks can be instantiated", {
  cb = WeightHistory$new()
  expect_is(cb, "KerasCallback")
  cb = auc_roc$new(training = as.list(1:2), validation = as.list(1:2))
  expect_is(cb, "KerasCallback")
  cb = KerasMetricsCallback_custom$new()
  expect_is(cb, "KerasCallback")
})


test_that("tfd families", {
  families =  c("normal", "bernoulli", "bernoulli_prob", "beta", "betar",
    "cauchy", "chi2", "chi",
    "exponential", "gamma", "gammar",
    "gumbel", "half_cauchy", "half_normal", "horseshoe",
    "inverse_gamma", "inverse_gaussian", "laplace",
    "log_normal", "logistic", "multinomial", "multinoulli", "negbinom",
    "pareto", "poisson", "poisson_lograte", "student_t", "student_t_ls",
    "uniform",
    "zip"
  )
  for (fam in families) {
    d = make_tfd_dist(fam)
    expect_is(d, "function")
    np = make_tfd_dist(fam, return_nrparams = TRUE)
    expect_true(np %in% c(1:3))
  }

    d = make_tfd_dist("zip", trafo_list = list(exp, exp))
    expect_is(d, "function")
})

