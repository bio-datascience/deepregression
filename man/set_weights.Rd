% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/methods.R
\name{set_weights}
\alias{set_weights}
\title{Function to set the weights of a deepregression object}
\usage{
set_weights(
  x,
  weights,
  param = NULL,
  type = c("linear", "nonlinear", "lasso", "ridge", "elasticnet")
)
}
\arguments{
\item{x}{deepregression object}

\item{weights}{a matrix with weights}

\item{param}{integer; for which parameter to set the weights}

\item{type}{character; for which type of layer to set the weights;}
}
\description{
Function to set the weights of a deepregression object
}
