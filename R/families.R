# helper funs tf
tfe <- function(x) tf$math$exp(x)
tfsig <- function(x) tf$math$sigmoid(x)
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
#'  \item{"beta": }{beta with concentration 1 = alpha (exp) and concentration 0 = beta (exp)}
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
#'  \item{"inverse_gaussian": }{inverse Gaussian with location (exp) and concentation (exp)}
#'  \item{"laplace": }{Laplace with location (identity) and scale (exp)}
#'  \item{"log_normal": }{Log-normal with location (identity) and scale (exp) of 
#'  underlying normal distribution}
#'  \item{"logistic": }{logistic with location (identity) and scale (exp)}
#'  \item{"negbinom": }{neg. binomial with mean (exp) and st.dev.(exp)}
#'  \item{"pareto": }{Pareto with concentration (exp) (and if modeled scale (exp), 
#'  else scale = 1)}
#'  \item{"poisson": }{poisson with rate (exp)}
#'  \item{"poisson_lograte": }{poisson with lograte (identity))}
#'  \item{"student_t": }{Student's t with df (exp)}
#'  \item{"student_t_ls": }{Student's t with df (exp), location (identity) and scale (exp)}
#'  \item{"uniform": }{uniform with upper and lower (both identity)}
#' }
#' 
#' @rdname dr_families
make_tfd_dist <- function(family, add_const = 1e-8, return_nrparams = FALSE)
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
                     inverse_gaussian = tfd_inverse_gaussian,
                     kumaraswamy = tfd_kumaraswamy,
                     laplace = tfd_laplace,
                     log_normal = tfd_log_normal,
                     logistic = tfd_logistic,
                     multinomial = tfd_multinomial,
                     negbinom = function(fail, probs) 
                       tfd_negative_binomial(total_count = fail, probs = probs,
                                             validate_args = TRUE),
                     pareto = tfd_pareto,
                     poisson = tfd_poisson,
                     poisson_lograte = function(log_rate) 
                       tfd_poisson(log_rate = log_rate),
                     student_t = function(x) 
                       tfd_student_t(df=x,loc=0,scale=1),
                     student_t_ls = tfd_student_t,
                     truncated_normal = tfd_truncated_normal,
                     uniform = tfd_uniform,
                     von_mises_fisher = tfd_von_mises_fisher,
                     von_mises = tfd_von_mises#,
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
                 "multinomial",
                 "truncated_normal",
                 "von_mises",
                 "von_mises_fisher",
                 "wishart",
                 "zipf"
  ) | grepl("multivariate", family) | grepl("vector", family))
  stop("Family ", family, " not implemented yet.")
  
  if(family=="binomial")
    stop("Family binomial not implemented yet. If you are trying to model independent",
         " draws from a bernoulli distribution, use family='bernoulli'.")
  
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
                       inverse_gaussian = list(function(x) add_const + tfe(x),
                                               function(x) add_const + tfe(x)),
                       kumaraswamy = list(), #tbd
                       laplace = list(function(x) x,
                                      function(x) add_const + tfe(x)),
                       log_normal = list(function(x) x,
                                         function(x) add_const + tfe(x)),
                       logistic = list(function(x) x,
                                       function(x) add_const + tfe(x)),
                       negbinom = list(function(x) x,
                                       function(x) x),
                       multinomial = list(), # tbd
                       pareto = list(function(x) add_const + tfe(x),
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
                       zipf = list(function(x) 1 + tfe(x))
  )
  
  
    
  ret_fun <- function(x) do.call(tfd_dist,
                                 lapply(1:ncol(x)[[1]],
                                        function(i)
                                          trafo_list[[i]](x[,i,drop=FALSE])))
  
  
  # return number of parameters if specified
  if(return_nrparams) return(length(trafo_list))
  
  return(ret_fun)    
  
}

family_trafo_funs <- function(family, add_const = 1e-8)
{
  
  # specially treated distributions
  trafo_fun <- switch(family,
    gammar = function(x){ 
      
      # rate = 1/((sigma^2)*mu)
      # con = (1/sigma^2)
      
      mu = add_const + tfe(x[,1,drop=FALSE])
      sig = add_const + tfe(x[,2,drop=FALSE])
      con = tfrec(tfsq(sig))
      rate = tfrec(tfmult(tfsq(sig),mu))
      
      return(list(con,rate))
    },
    betar = function(x){
      
      # mu=a/(a+b) 
      # sig=(1/(a+b+1))^0.5
      mu = tfsig(x[,1,drop=FALSE])
      sigsq = tfsq(tfsig(x[,2,drop=FALSE]))
      a = tfmult(mu, (tfrec(sigsq) - 1))
      b = tfmult((tfrec(mu) - 1), a)
      
      return(list(a,b))
    },
    negbinom = function(x){
      
      # see, e.g., https://www.johndcook.com/negative_binomial.pdf
      
      mu <- tfe(x[,1,drop=FALSE])
      sig2 <- tfsq(tfe(x[,2,drop=FALSE]))
      f = tf$compat$v2$clip_by_value(
        tfdiv(tfsq(mu),sig2-mu), 
        add_const, Inf
        )
      p = tf$compat$v2$clip_by_value(
        tfdiv(f, f+mu), 
        0, 1
        )
      
      return(list(f,p))
    }
  )
      
    return(trafo_fun)
    
}