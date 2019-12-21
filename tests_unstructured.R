################################## Tests #####################################
silent = TRUE
#####################################################################
#############
############# Formulae / Model specification #############
#############

set.seed(24)

# generate the data
n <- 1500
b0 <- 1

# training data; predictor 
x <- runif(n) %>% as.matrix()
z <- runif(n)
fac <- gl(10, n/10)

true_mean_fun <- function(xx) sin(10*xx) + b0

# training data
y <- true_mean_fun(x) + rnorm(n = n, mean = 0, sd = 2)

data = data.frame(x = x, fac = fac, z = z)

# test data
x_test <- runif(n) %>% as.matrix()

validation_data = data.frame(x = x_test)

y_test <- true_mean_fun(x_test) + rnorm(n = n, sd = 2)

deep_model <- function(x) x %>% 
  layer_dense(units = 4, activation = "relu") %>% 
  layer_dense(units = 1, activation = "linear")

# first without the need for orthogonalization
formulae <- c(
  "~ 0 + x",
  "~ 1 + x",
  "~ 1 + x + z",
  "~ 0 + s(x)",
  "~ 1 + s(x)",
  "~ 1 + s(x) + s(z)",
  "~ 1 + te(x,z)",
  "~ 1 + d(x) + z",
  "~ 1 + d(x,z)",
  "~ 1 + d(x) + s(z)",
  "~ 1 + s(x) + fac",
  "~ 1 + d(x) + fac",
  "~ 1 + d(x) + s(z,by=fac)"
)

for(form in formulae){

  cat("Formula: ", form, " ... ")
  suppressWarnings(
    mod <- try(deepregression(
      y = y,
      data = data,
      # define how parameters should be modeled
      list_of_formulae = list(loc = as.formula(form), scale = ~1),
      list_of_deep_models = list(deep_model)
    ), silent=silent)
  )
  # test if model can be fitted
  if(class(mod)=="try-error")
  {
    cat("Failed to initialize the model.\n")
    next
  }
  fitting <- try(
    res <- mod %>% fit(epochs=2, verbose = FALSE, view_metrics = FALSE),
    silent=silent
  )
  if(class(fitting)=="try-error"){ 
    cat("Failed to fit the model.\n")
  }else{
    # print(res$metrics)
    cat("Success.\n")
  }
}


#############
############# CV: #############
#############

for(form in formulae){
  
  cat("Formula: ", form, " ... ")
  suppressWarnings(
    mod <- try(deepregression(
      y = y,
      data = data,
      # define how parameters should be modeled
      list_of_formulae = list(loc = as.formula(form), scale = ~1),
      list_of_deep_models = list(deep_model),
      cv_folds = 2
    ), silent=silent)
  )
  # test if model can be fitted
  if(class(mod)=="try-error")
  {
    cat("Failed to initialize the model.\n")
    next
  }
  fitting <- try(
    res <- mod %>% cv(epochs=3, print_folds = FALSE),
    silent=silent
  )
  if(class(fitting)[1]=="try-error"){ 
    cat("Failed to cross-validate the model.\n")
  }else{
    # print(res$metrics)
    cat("Success.\n")
  }
}

#############
############# Array Inputs: #############
#############

mnist <- dataset_mnist()

train_X <- list(x=array(mnist$train$x, 
                        # so that we can use 2d conv later
                        c(dim(mnist$train$x),1))
)
subset <- 1:1000
train_X[[1]]<- train_X[[1]][subset,,,,drop=FALSE]
train_y <- to_categorical(mnist$train$y[subset])

conv_mod <- function(x) x %>% 
  layer_conv_2d(filters = 16, kernel_size = c(3,3),
                activation= "relu",
                input_shape = shape(NULL, NULL, 1)) %>% 
  layer_global_average_pooling_2d() %>% 
  layer_dense(units = 10)

simple_mod <- function(x) x %>% 
  layer_dense(units = 4, activation = "relu") %>% 
  layer_dense(units = 1, activation = "linear")

z <- rnorm(length(subset))
fac <- gl(4, length(subset)/4)
m <- runif(length(z))

list_as_input <- append(train_X, (data.frame(z=z, fac=fac, m=m)))

mod <- deepregression(y = train_y, list_of_formulae = 
                        list(logit = ~ 1 + simple_mod(z) + fac + conv_mod(x)), 
                      data = list_as_input,
                      list_of_deep_models = list(simple_mod = simple_mod,
                                                 conv_mod = conv_mod),
                      family = "multinoulli")

cvres <- mod %>% cv(epochs = 2, cv_folds = 2, batch_size=1)

mod %>% fit(epochs = 2, 
            batch_size=1, #steps_per_epoch=1, 
            view_metrics=FALSE,
            validation_split = NULL)


#############
############# Deep Specification: #############
#############

k <- rnorm(length(x))
data$k <- k

another_deep_model <- function(x) x %>% 
  layer_dense(units = 4, activation = "relu") %>% 
  layer_dense(units = 1, activation = "linear")

third_model <- function(x) x %>% 
  layer_dense(units = 4, activation = "relu") %>% 
  layer_dense(units = 1, activation = "linear")

# first without the need for orthogonalization
formulae <- c(
  "~ d(x,z) + k",
  "~ d(x,z,k)",
  "~ d(x) + d(z)",
  "~ deep_model(x) + another_deep_model(z)",
  "~ deep_model(x,z) + another_deep_model(k)",
  "~ deep_model(x) + another_deep_model(z) + third_model(k)"
)

list_models <- list(deep_model = deep_model,
                    another_deep_model = another_deep_model,
                    third_model = third_model)

use <- list(1,1,1:2,1:2,1:2,1:3)

for(i in 1:length(formulae)){
  
  form <- formulae[i]
  usei <- use[[i]]
  this_list <- list_models[usei]
  if(length(usei)==1) names(this_list) <- NULL
  
  cat("Formula: ", form, " ... ")
  suppressWarnings(
    mod <- try(deepregression(
      y = y,
      data = data,
      # define how parameters should be modeled
      list_of_formulae = list(loc = as.formula(form), scale = ~1),
      list_of_deep_models = this_list
    ), silent=silent)
  )
  # test if model can be fitted
  if(class(mod)=="try-error")
  {
    cat("Failed to initialize the model.\n")
    next
  }
  fitting <- try(
    res <- mod %>% fit(epochs=2, verbose = FALSE, view_metrics = FALSE),
    silent=silent
  )
  if(class(fitting)=="try-error"){ 
    cat("Failed to fit the model.\n")
  }else{
    # print(res$metrics)
    cat("Success.\n")
  }
}


#############
############# Prediction: #############
#############

#############
############# Plotting: #############
#############


#############
############# Families: #############
#############

set.seed(24)

# generate the data
n <- 1500
b0 <- 1

# training data; predictor 
x <- runif(n) %>% as.matrix()
z <- runif(n) %>% as.matrix()
y <- runif(n) %>% as.matrix()
data = data.frame(x = x, z = z)

dists = 
  c(
    "normal", "bernoulli", "bernoulli_prob", 
    "beta", "betar", "cauchy", "chi2", "chi","exponential",
    "gamma", "gammar", "gumbel", "half_cauchy", "half_normal", "horseshoe",
    "inverse_gamma", "inverse_gaussian", "laplace", "log_normal",
    "logistic", "negbinom", "negbinom", "pareto", 
    "poisson", "poisson_lograte", "student_t",
    "student_t_ls", "uniform"
  )

for(dist in dists)
{
  cat("Fitting", dist, "model... ")
  suppressWarnings(
    mod <- try(deepregression(
      y = y,
      data = data,
      # define how parameters should be modeled
      list_of_formulae = list(~ 1 + x, ~ 1 + z, ~ 1),
      list_of_deep_models = NULL,
      family = dist
    ), silent=silent)
  )
  # test if model can be fitted
  if(class(mod)=="try-error")
  {
    cat("Failed to initialize the model.\n")
    next
  }
  fitting <- try(
    res <- mod %>% fit(epochs=2, verbose = FALSE, view_metrics = FALSE),
    silent=silent
  )
  if(class(fitting)=="try-error"){ 
    cat("Failed to fit the model.\n")
  }else if(sum(is.nan(unlist(res$metrics))) > 0){
    cat("NaNs in loss or validation loss.\n")
  }else if(any(unlist(res$metrics)==Inf)){
    cat("Infinite values in loss or validation loss.\n")
  }else{
    # print(res$metrics)
    cat("Success.\n")
  }
}

#############
############# Orthogonalization: #############
#############

set.seed(24)

n <- 150
ps <- c(1,3,5)
b0 <- 1
simnr <- 10
true_sd <- 2

deep_model <- function(x) x %>% 
  layer_dense(units = 32, activation = "relu") %>% 
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 16, activation = "relu") %>% 
  layer_dense(units = 1, activation = "linear")

list_of_funs <-  list(function(x) sin(10*x),
                      function(x) tanh(3*x),
                      function(x) x^2,
                      function(x) cos(x*3-2)*(-x*3),
                      function(x) exp(x*2) - 1
)

for(p in 1:5){
  
  X <- matrix(runif(p*n), ncol=p)
  partpred_l <- sapply(1:p, function(j) 4/j*X[,j])
  partpred_nl <- sapply(1:p, function(j)
    list_of_funs[[j]](X[,j]))
  
  true_mean <- b0 + rowSums(partpred_l) + rowSums(partpred_l)
  
  # training data
  y <- true_mean + rnorm(n = n, mean = 0, sd = true_sd)
  
  data = data.frame(X)
  colnames(data) <- paste0("V", 1:p)
  
  #####################################################################
  vars <- paste0("V", 1:p)
  form <- paste0("~ 1 + ", paste(vars, collapse = " + "), " + s(",
                 paste(vars, collapse = ") + s("), ") + d(",
                 paste(vars, collapse = ", "), ")")
  
  cat("Fitting model with ", p, "orthogonalization(s) ... ")
  #####################################################################
  suppressWarnings(
    mod <- try(deepregression(
    # supply data (response and data.frame for covariates)
    y = y,
    data = data,
    # define how parameters should be modeled
    list_of_formulae = list(loc = as.formula(form), scale = ~1),
    list_of_deep_models = list(deep_model),
    cv_folds = 5
    ), silent=silent)
  )
  # test if model can be fitted
  if(class(mod)=="try-error")
  {
    cat("Failed to initialize the model.\n")
    next
  }
  fitting <- try(
    res <- mod %>% fit(epochs=2, verbose = FALSE, view_metrics = FALSE),
    silent=silent
  )
  if(class(fitting)=="try-error"){ 
    cat("Failed to fit the model.\n")
  }else{
    # print(res$metrics)
    cat("Success.\n")
  }
}
