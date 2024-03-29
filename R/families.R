# helper funs tf
tfe <- function(x) tf$math$exp(x)
tfsig <- function(x) tf$math$sigmoid(x)
tfsoft <- function(x) tf$math$softmax(x)
tfsqrt <- function(x) tf$math$sqrt(x)
tfsq <- function(x) tf$math$square(x)
tfdiv <- function(x,y) tf$math$divide(x,y)
tfrec <- function(x) tf$math$reciprocal(x)
tfmult <- function(x,y) tf$math$multiply(x,y)

#' Families for deepregression
#'
#' @param family character vector
#'
#' @details
#' To specify a custom distribution, define the a function as follows
#' \code{
#' function(x) do.call(your_tfd_dist, lapply(1:ncol(x)[[1]],
#'                                     function(i)
#'                                      your_trafo_list_on_inputs[[i]](
#'                                        x[,i,drop=FALSE])))
#' }
#' and pass it to \code{deepregression} via the \code{dist_fun} argument.
#' Currently the following distributions are supported
#' with parameters (and corresponding inverse link function in brackets):
#'
#' \itemize{
#'  \item{"normal": }{normal distribution with location (identity), scale (exp)}
#'  \item{"bernoulli": }{bernoulli distribution with logits (identity)}
#'  \item{"bernoulli_prob": }{bernoulli distribution with probabilities (sigmoid)}
#'  \item{"beta": }{beta with concentration 1 = alpha (exp) and concentration
#'  0 = beta (exp)}
#'  \item{"betar": }{beta with mean (sigmoid) and scale (sigmoid)}
#'  \item{"cauchy": }{location (identity), scale (exp)}
#'  \item{"chi2": }{cauchy with df (exp)}
#'  \item{"chi": }{cauchy with df (exp)}
#'  \item{"exponential": }{exponential with lambda (exp)}
#'  \item{"gamma": }{gamma with concentration (exp) and rate (exp)}
#'  \item{"gammar": }{gamma with location (exp) and scale (exp)}
#'  \item{"gumbel": }{gumbel with location (identity), scale (exp)}
#'  \item{"half_cauchy": }{half cauchy with location (identity), scale (exp)}
#'  \item{"half_normal": }{half normal with scale (exp)}
#'  \item{"horseshoe": }{horseshoe with scale (exp)}
#'  \item{"inverse_gamma": }{inverse gamma with concentation (exp) and rate (exp)}
#'  \item{"inverse_gamma_ls": }{inverse gamma with location (exp) and variance (1/exp)}
#'  \item{"inverse_gaussian": }{inverse Gaussian with location (exp) and concentation
#'  (exp)}
#'  \item{"laplace": }{Laplace with location (identity) and scale (exp)}
#'  \item{"log_normal": }{Log-normal with location (identity) and scale (exp) of
#'  underlying normal distribution}
#'  \item{"logistic": }{logistic with location (identity) and scale (exp)}
#'  \item{"negbinom": }{neg. binomial with count (exp) and prob (sigmoid)}
#'  \item{"negbinom_ls": }{neg. binomail with mean (exp) and clutter factor (exp)}
#'  \item{"pareto": }{Pareto with concentration (exp) and scale (1/exp)} 
#'  \item{"pareto_ls": }{Pareto location scale version with mean (exp) 
#'  and scale (exp), which corresponds to a Pareto distribution with parameters scale = mean
#'  and concentration = 1/sigma, where sigma is the scale in the pareto_ls version.}
#'  \item{"poisson": }{poisson with rate (exp)}
#'  \item{"poisson_lograte": }{poisson with lograte (identity))}
#'  \item{"student_t": }{Student's t with df (exp)}
#'  \item{"student_t_ls": }{Student's t with df (exp), location (identity) and
#'  scale (exp)}
#'  \item{"uniform": }{uniform with upper and lower (both identity)}
#'  \item{"zinb": }{Zero-inflated negative binomial with mean (exp), 
#'  variance (exp) and prob (sigmoid)}
#'  \item{"zip":  }{Zero-inflated poisson distribution with mean (exp) and prob (sigmoid)}
#' }
#' @param add_const small positive constant to stabilize calculations
#' @param return_nrparams logical, if TRUE, only the number of distribution parameters is
#' returned; else (FALSE) the \code{dist_fun} required in \code{deepregression}
#' @param trafo_list list of transformations for each distribution parameter.
#' Per default the transformation listed in details is applied.
#'
#' @export
#' @rdname dr_families
make_tfd_dist <- function(family, add_const = 1e-8,
                          return_nrparams = FALSE, trafo_list = NULL)
{

  # define dist_fun
  tfd_dist <- switch(family,
                     normal = tfd_normal,
                     bernoulli = tfd_bernoulli,
                     bernoulli_prob = function(probs)
                       tfd_bernoulli(probs = probs),
                     beta = tfd_beta,
                     betar = tfd_beta,
                     binomial = tfd_binomial,
                     categorical = tfd_categorical,
                     cauchy = tfd_cauchy,
                     chi2 = tfd_chi2,
                     chi = tfd_chi,
                     dirichlet_multinomial = tfd_dirichlet_multinomial,
                     dirichlet = tfd_dirichlet,
                     exponential = tfd_exponential,
                     gamma_gamma = tfd_gamma_gamma,
                     gamma = tfd_gamma,
                     gammar = tfd_gamma,
                     geometric = tfd_geometric,
                     gumbel = tfd_gumbel,
                     half_cauchy = tfd_half_cauchy,
                     half_normal = tfd_half_normal,
                     horseshoe = tfd_horseshoe,
                     inverse_gamma = tfd_inverse_gamma,
                     inverse_gamma_ls = tfd_inverse_gamma,
                     inverse_gaussian = tfd_inverse_gaussian,
                     kumaraswamy = tfd_kumaraswamy,
                     laplace = tfd_laplace,
                     log_normal = tfd_log_normal,
                     logistic = tfd_logistic,
                     multinomial = function(probs)
                       tfd_multinomial(total_count = 1L,
                                       probs = probs),
                     multinoulli = function(logits)#function(probs)
                       # tfd_multinomial(total_count = 1L,
                       #                 logits = logits),
                       tfd_one_hot_categorical(logits),
                       # tfd_categorical,#(probs = probs),
                     negbinom = function(fail, probs)
                       tfd_negative_binomial(total_count = fail, probs = probs#,
                                             # validate_args = TRUE
                       ),
                     negbinom_ls = tfd_negative_binomial_ls,
                     pareto = tfd_pareto,
                     pareto_ls = tfd_pareto,
                     poisson = tfd_poisson,
                     poisson_lograte = function(log_rate)
                       tfd_poisson(log_rate = log_rate),
                     student_t = function(x)
                       tfd_student_t(df=x,loc=0,scale=1),
                     student_t_ls = tfd_student_t,
                     truncated_normal = tfd_truncated_normal,
                     uniform = tfd_uniform,
                     von_mises_fisher = tfd_von_mises_fisher,
                     von_mises = tfd_von_mises,
                     zinb = tfd_zinb,
                     zip = tfd_zip
                     # zipf = function(x)
                     #   tfd_zipf(x,
                     #            dtype = tf$float32,
                     #            sample_maximum_iterations =
                     #              tf$constant(100, dtype="float32"))
  )

  # families not yet implemented
  if(family%in%c("categorical",
                 "dirichlet_multinomial",
                 "dirichlet",
                 "gamma_gamma",
                 "geometric",
                 "kumaraswamy",
                 "truncated_normal",
                 "von_mises",
                 "von_mises_fisher",
                 "wishart",
                 "zipf"
  ) | grepl("multivariate", family) | grepl("vector", family))
  stop("Family ", family, " not implemented yet.")

  if(family=="binomial")
    stop("Family binomial not implemented yet.",
         " If you are trying to model independent",
         " draws from a bernoulli distribution, use family='bernoulli'.")
  if(is.null(trafo_list))
    trafo_list <- switch(family,
                         normal = list(function(x) x,
                                       function(x) add_const + tfe(x)),
                         bernoulli = list(function(x) x),
                         bernoulli_prob = list(function(x) tfsig(x)),
                         beta = list(function(x) add_const + tfe(x),
                                     function(x) add_const + tfe(x)),
                         betar = list(function(x) x,
                                      function(x) x),
                         binomial = list(), # tbd
                         categorial = list(), #tbd
                         cauchy = list(function(x) x,
                                       function(x) add_const + tfe(x)),
                         chi2 = list(function(x) add_const + tfe(x)),
                         chi = list(function(x) add_const + tfe(x)),
                         dirichlet_multinomial = list(), #tbd
                         dirichlet = list(), #tbd
                         exponential = list(function(x) add_const + tfe(x)),
                         gamma_gamma = list(), #tbd
                         gamma = list(function(x) add_const + tfe(x),
                                      function(x) add_const + tfe(x)),
                         geometric = list(function(x) x),
                         gammar = list(function(x) x,
                                       function(x) x),
                         gumbel = list(function(x) x,
                                       function(x) add_const + tfe(x)),
                         half_cauchy = list(function(x) x,
                                            function(x) add_const + tfe(x)),
                         half_normal = list(function(x) add_const + tfe(x)),
                         horseshoe = list(function(x) add_const + tfe(x)),
                         inverse_gamma = list(function(x) add_const + tfe(x),
                                              function(x) add_const + tfe(x)),
                         inverse_gamma_ls = list(function(x) add_const + tfe(x),
                                              function(x) add_const + tfe(x)),
                         inverse_gaussian = list(function(x) add_const + tfe(x),
                                                 function(x)
                                                   add_const + tfe(x)),
                         kumaraswamy = list(), #tbd
                         laplace = list(function(x) x,
                                        function(x) add_const + tfe(x)),
                         log_normal = list(function(x) x,
                                           function(x) add_const + tfe(x)),
                         logistic = list(function(x) x,
                                         function(x) add_const + tfe(x)),
                         negbinom = list(function(x) add_const + tfe(x),
                                         function(x) tf$math$sigmoid(x)),
                         negbinom_ls = list(function(x) add_const + tfe(x),
                                            function(x) add_const + tfe(x)),
                         multinomial = list(function(x) tfsoft(x)),
                         multinoulli = list(function(x) x),
                         pareto = list(function(x) add_const + tfe(x),
                                       function(x) add_const + tfe(-x)),
                         pareto_ls = list(function(x) add_const + tfe(x),
                                       function(x) add_const + tfe(x)),
                         poisson = list(function(x) add_const + tfe(x)),
                         poisson_lograte = list(function(x) x),
                         student_t = list(function(x) x),
                         student_t_ls = list(function(x) add_const + tfe(x),
                                             function(x) x,
                                             function(x) add_const + tfe(x)),
                         truncated_normal = list(), # tbd
                         uniform = list(function(x) x,
                                        function(x) x),
                         von_mises = list(function(x) x,
                                          function(x) add_const + tfe(x)),
                         zinb = list(function(x) add_const + tfe(x),
                                     function(x) add_const + tfe(x),
                                     function(x) tf$stack(list(tf$math$sigmoid(x),
                                                               1-tf$math$sigmoid(x)),
                                                          axis=2L)),
                         zip = list(function(x) add_const + tfe(x),
                                    function(x) tf$stack(list(tf$math$sigmoid(x),
                                                              1-tf$math$sigmoid(x)),
                                                         axis=2L)),
                         zipf = list(function(x) 1 + tfe(x))
    )


  ret_fun <- function(x) do.call(tfd_dist,
                                 lapply(1:ncol(x)[[1]],
                                        function(i)
                                          trafo_list[[i]](
                                            x[,i,drop=FALSE])))

  if(family=="multinomial"){

    ret_fun <- function(x) tfd_dist(trafo_list[[1]](x))

  }
  if(family=="multinoulli"){

    ret_fun <- function(x) tfd_dist(trafo_list[[1]](x))

  }

  # return number of parameters if specified
  if(return_nrparams) return(length(trafo_list))

  return(ret_fun)

}

names_families <- function(family)
{
  
  nams <- switch(family,
                 normal = c("location", "scale"),
                 bernoulli = "logits",
                 bernoulli_prob = "probabilities",
                 beta = c("concentration", "concentration"),
                 betar = c("location", "scale"),
                 cauchy = c("location", "scale"),
                 chi2 = "df",
                 chi = "df",
                 exponential = "rate",
                 gamma = c("concentration", "rate"),
                 gammar = c("location", "scale"),
                 gumbel = c("location", "scale"),
                 half_cauchy = c("location", "scale"),
                 half_normal = "scale",
                 horseshoe = "scale",
                 inverse_gamma = c("concentation", "rate"),
                 inverse_gamma_ls = c("location", "scale"),
                 inverse_gaussian = c("location", "concentation"),
                 laplace = c("location", "scale"),
                 log_normal = c("location", "scale"),
                 logistic = c("location", "scale"),
                 negbinom = c("count", "prob"),
                 negbinom_ls = c("mean", "clutter_factor"),
                 pareto = c("concentration", "scale"),
                 pareto_ls = c("location", "scale"),
                 poisson = "rate",
                 poisson_lograte = "lograte",
                 student_t = "df",
                 student_t_ls = c("df", "location", "scale"),
                 uniform = c("upper", "lower"),
                 zinb = c("mean", "variance", "prob"),
                 zip = c("mean", "prob")
  )
  
  return(nams)
  
}

family_trafo_funs_special <- function(family, add_const = 1e-8)
{

  # specially treated distributions
  trafo_fun <- switch(family,
    gammar = function(x){

      # rate = 1/((sigma^2)*mu)
      # con = (1/sigma^2)

      mu = tfe(x[,1,drop=FALSE])
      sig = tfe(x[,2,drop=FALSE])
      # con = #tf$compat$v2$maximum(
      #   tfrec(tfsq(sig))#,0 + add_const)
      # rate = #tf$compat$v2$maximum(
      #   tfrec(tfmult(tfsq(sig),mu))#,0 + add_const)
      sigsq = tfsq(sig)
      con = tfdiv(tfsq(mu), sigsq) + add_const
      rate = tfdiv(mu, sigsq) + add_const

      # rate = tfrec(tfe(x[,2,drop=FALSE]))
      # con = tfdiv(tfe(x[,1,drop=FALSE]), tfe(x[,1,drop=FALSE]))

      return(list(concentration = con, rate = rate))
    },
    betar = function(x){

      # mu=a/(a+b)
      # sig=(1/(a+b+1))^0.5
      mu = tfsig(x[,1,drop=FALSE])
      sigsq = tfsq(tfsig(x[,2,drop=FALSE]))
      #a = tf$compat$v2$maximum(tfmult(mu, (tfrec(sigsq) - 1)), 1 + add_const)
      #b = tf$compat$v2$maximum(tfmult((tfrec(mu) - 1), a), 1 + add_const)
      a = tf$compat$v2$maximum(
        tfmult(mu, tfdiv(tf$constant(1) - sigsq, sigsq)),
        tf$constant(0) + add_const)
      b = tf$compat$v2$maximum(
        tfmult(a, tfdiv(tf$constant(1) - mu,mu)),
        tf$constant(0) + add_const)

      return(list(concentration1 = a, concentration0 = b))
    },
    pareto_ls = function(x){
      
      # k_print_tensor(x, message = "This is x")
      scale = add_const + tfe(x[,1,drop=FALSE])
      # k_print_tensor(scale, message = "This is scale")
      con = tfe(-x[,2,drop=FALSE])
      # k_print_tensor(con, message = "This is con")
      return(list(concentration = con, scale = scale)) 
      
      
    },
    inverse_gamma_ls = function(x){
      
      # alpha = 1/sigma^2
      alpha = add_const + tfe(-x[,2,drop=FALSE])
      # beta = mu (alpha + 1)
      beta = add_const + tfe(x[,1,drop=FALSE]) * (alpha + 1)
      
      return(list(concentration = alpha, scale = beta)) 
      
      
    }
  )

  tfd_dist <- switch(family,
                     betar = tfd_beta,
                     gammar = tfd_gamma,
                     pareto_ls = tfd_pareto,
                     inverse_gamma_ls = tfd_inverse_gamma
  )

  ret_fun <- function(x) do.call(tfd_dist, trafo_fun(x))

  return(ret_fun)

}

#' generate mixture distribution of same family
#'
#' @param dist tfp distribution
#' @param nr_comps number of mixture components
#' @param trafos_each_param list of transformaiton applied before plugging
#' the linear predictor into the parameters of the distributions.
#' Should be of length #parameters of \code{dist}
#' @return returns function than can be used as argument \code{dist\_fun} for
#' @export
#'
mix_dist_maker <- function(
  dist = tfd_normal,
  nr_comps = 3,
  trafos_each_param = list(function(x) x, function(x) 1e-8 + tfe(x))
  ){

  stack <- function(x,ind=1:nr_comps) tf$stack(
    lapply(ind, function(j)
      x[,j,drop=FALSE]), 2L)

  mixdist = function(probs, params)
  {

    mix = tfd_categorical(probs = probs)
    this_components = do.call(dist, params)

    res_dist <- tfd_mixture_same_family(
      mixture_distribution=mix,
      components_distribution=this_components
    )

    return(res_dist)
  }

  trafo_fun <- function(x){

    c(probs = list(stack(x,1:nr_comps)),
      params = list(
        lapply(1:length(trafos_each_param),
               function(i)
                 stack(trafos_each_param[[i]](
                   x[, nr_comps +
                       # first x for pis
                       (i-1)*nr_comps +
                       # then for each parameter
                       (1:nr_comps),drop=FALSE]
                 )
                 )
        )
      )
    )

  }

  return(
    function(x) do.call(mixdist, trafo_fun(x))
  )

}

#' Implementation of a zero-inflated poisson distribution for TFP
#'
#' @param lambda scalar value for rate of poisson distribution
#' @param probs vector of probabilites of length 2 (probability for poisson and
#' probability for 0s)
tfd_zip <- function(lambda, probs)
{

  return(
    tfd_mixture(cat = tfd_categorical(probs = probs),
                components =
                  list(tfd_poisson(rate = lambda),
                       tfd_deterministic(loc = lambda * 0L)
                  ),
                name="zip")
  )
}

tfd_negative_binomial_ls = function(mu, r){

  # sig2 <- mu + (mu*mu / r)
  # count <- r
  probs <- #1-tf$compat$v2$clip_by_value(
    r / (r + mu)#,
    # 0, 1
  # )
  
  return(tfd_negative_binomial(total_count = r, probs = probs))

}

#' Implementation of a zero-inflated negbinom distribution for TFP
#'
#' @param mu,r parameter of the negbin_ls distribution
#' @param probs vector of probabilites of length 2 (probability for poisson and
#' probability for 0s)
tfd_zinb <- function(mu, r, probs)
{

  return(
    tfd_mixture(cat = tfd_categorical(probs = probs),
                components =
                  list(tfd_negative_binomial_ls(mu = mu, r = r),
                       tfd_deterministic(loc = mu * 0L)
                  ),
                name="zinb")
  )
}


#' Implementation of a multivariate normal distribution
#'
#' @param dim dimension of the multivariate normal distribution
#' @param with_cov logical; whether or not to have a full covariance
#' @param trafos_scale transformation function for the scale
#' @param add_const small positive constant to stabilize calculations
multinorm_maker <- function(dim = 2,
                            with_cov = TRUE,
                            trafos_scale = exp,
                            add_const = 1e-8)
{

  nr_cov_low_tril <- dim * (dim - 1) / 2 + dim

  dims_to_cov <- function(ind)
  {

    temp <- matrix(1:dim^2,ncol=dim,byrow = T)
    res <- ind[order(c(diag(temp),temp[upper.tri(temp,diag=F)],temp[lower.tri(temp,diag=F)]))]
    res[is.na(res)] <- ind[1] # will be ignored anyway
    return(res)

  }

  ind_diag <- seq(1,dim^2,by=dim+1)

  scale_tr <- function(x) add_const + tfe(x)

  if(with_cov){

    trafo_fun <- function(x){

      ind_cov <- dims_to_cov((dim+1):(dim+nr_cov_low_tril))

      p1 <- scale_tr(x[,(dim+1):(2*dim),drop=FALSE])
      p2 <- x[,(2*dim+1):(dim+nr_cov_low_tril),drop=FALSE]
      mat <- tf$concat(list(p1,p2),1L)
      mat <- tf$concat(lapply(ind_cov-dim, function(i) mat[,i,drop=FALSE]),1L)
      # mat[,ind_cov-dim,drop=FALSE]
      expmat <- tf$expand_dims(mat, 2L)
      resexpmat <- tf$reshape(expmat, shape = list(as.integer(tf$shape(expmat)[1]),
                                                   as.integer(dim),
                                                   as.integer(dim)))

      c(loc = list(x[,1:dim,drop=FALSE]),
        scale_tril = resexpmat
      )

    }

    return(
      function(x) do.call(tfd_multivariate_normal_tri_l,
                          trafo_fun(x))
    )

  }else{

    trafo_fun <- function(x){

      list(loc = x[,1:dim,drop=FALSE],
           scale_diag = add_const + tfe(x[,(dim+1):(dim*2),drop=FALSE])
      )

    }

    return(
      function(x) do.call(tfd_multivariate_normal_diag,
                          trafo_fun(x))
    )

  }

}
