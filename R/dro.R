# from mboost https://github.com/cran/mboost/blob/master/R/helpers.R
make_psd <- function(x, eps = sqrt(.Machine$double.eps)) {
  lambda <- min(eigen(x, only.values = TRUE, symmetric = TRUE)$values)
  ## use some additional tolerance to ensure semipositive definite matrices
  lambda <- lambda - sqrt(.Machine$double.eps)
  # smallest eigenvalue negative = not semipositive definite
  if (lambda < -1e-10) {
    rho <- 1/(1-lambda)
    x <- rho * x + (1-rho) * diag(dim(x)[1])
    ## now check if it is really positive definite by recursively calling
    x <- make_psd(x)
  }
  return(x)
}

# this is a tweaked version of df2lambda from mboost
# which is a huge bottleneck for model initialization
# if dmat is large due to the used SVD
# => we allow for an approximation based on the package rsvd
DRO <- function(X, df = 4, lambda = NULL, dmat = NULL, # weights,
                svdtype = c("default", "rsvd"), XtX = NULL,
                k = 100, q = 3, hat1 = TRUE, ...) {

  svdtype <- match.arg(svdtype)
  if(svdtype=="rsvd") svd <- function(A, nu, nv) rsvd::rsvd(A = A, nu = nu, nv = nv,
                                                            k = 100, q = 3, ...)

  stopifnot(xor(is.null(df), is.null(lambda)))
  if (!is.null(df)) {
    rank_X <- rankMatrix(X, method = 'qr', warn.t = FALSE)
    if (df >= rank_X) {
      if (df > rank_X)
        warning(sQuote("df"),
                " too large:\n  Degrees of freedom cannot be larger",
                " than the rank of the design matrix.\n",
                "  Unpenalized base-learner with df = ",
                rank_X, " used. Re-consider model specification.")
      return(c(df = df, lambda = 0))
    }
  }
  if (!is.null(lambda))
    if (lambda == 0)
      return(c(df = rankMatrix(X), lambda = 0))

  # Demmler-Reinsch Orthogonalization (cf. Ruppert et al., 2003,
  # Semiparametric Regression, Appendix B.1.1).

  ### there may be more efficient ways to compute XtX, but we do this
  ### elsewhere (e.g. in %O%)
  if (is.null(XtX))
    XtX <- crossprod(X) # * sqrt(weights))
  if (is.null(dmat)) {
    if(is(XtX, "Matrix")) diag <- Diagonal
    dmat <- diag(ncol(XtX))
  }
  ## avoid that XtX matrix is not (numerically) singular
  A <- XtX + dmat * 1e-09
  ## make sure that A is also numerically positiv semi-definite
  A <- make_psd(as.matrix(A))
  ## make sure that A is also numerically symmetric
  if (is(A, "Matrix"))
    A <- forceSymmetric(A)
  Rm <- backsolve(chol(A), x = diag(ncol(XtX)))
  ## singular value decomposition without singular vectors
  d <- try(svd(crossprod(Rm, dmat) %*% Rm, nu=0, nv=0)$d)
  ## if unsucessfull try the same computation but compute singular vectors as well
  if (inherits(d, "try-error"))
    d <- svd(crossprod(Rm, dmat) %*% Rm)$d
  if (hat1) {
    dfFun <- function(lambda) sum(1/(1 + lambda * d))
  }
  else {
    dfFun <- function(lambda) 2 * sum(1/(1 + lambda * d)) -
      sum(1/(1 + lambda * d)^2)
  }
  if (!is.null(lambda))
    return(c(df = dfFun(lambda), lambda = lambda))
  if (df >= length(d)) return(c(df = df, lambda = 0))

  # search for appropriate lambda using uniroot
  df2l <- function(lambda)
    dfFun(lambda) - df

  lambdaMax <- 1e+16

  if (df2l(lambdaMax) > 0){
    if (df2l(lambdaMax) > sqrt(.Machine$double.eps))
    return(c(df = df, lambda = lambdaMax))
  }
  lambda <- uniroot(df2l, c(0, lambdaMax), tol = sqrt(.Machine$double.eps))$root
  if (abs(df2l(lambda)) > sqrt(.Machine$double.eps))
    warning("estimated degrees of freedom differ from ", sQuote("df"),
            " by ", df2l(lambda))
  return(c(df = df, lambda = lambda))
}
