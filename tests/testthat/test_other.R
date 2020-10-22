context("check and install")

test_that("check_and_install", {
  expect_message(check_and_install(), "Tensorflow found, skipping tensorflow installation!")
})
