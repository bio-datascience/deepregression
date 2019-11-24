# function that extracts variables from special symbols in formulae
extract_from_special <- function(x)
  trimws(
    strsplit(regmatches(x, gregexpr("(?<=\\().*?(?=\\))", x, perl=T))[[1]],
             split = ",")[[1]]
  )
# convert sparse matrix to sparse tensor
sparse_mat_to_tensor <- function(X)
{

  missing_ind <- setdiff(c("i","j","p"), slotNames(X))
  if(missing_ind=="j")
    j = findInterval(seq(X@x)-1,X@p[-1])
  if(missing_ind=="i") stop("Sparse Matrix with missing i not implemented yet.")
  i = X@i
  tf$SparseTensor(indices = lapply(1:length(i), function(ind) c(i[ind], j[ind])),
                  values = X@x,
                  dense_shape = X@Dim)

}

NCOL0 <- function(x)
{
  if(is.null(x))
    return(0)
  return(NCOL(x))
}

# get contents from formula
get_contents <- function(lf, data, df, variable_names, intercept = TRUE, defaultSmoothing){
  # extract which parts are modelled as deep parts
  # which by smooths, which linear
  specials <- c("s", "te", "ti", "d")
  tf <- terms.formula(lf, specials=specials)
  if(length(attr(tf, "term.labels"))==0){
    if(intercept & attr(tf,"intercept")){
      linterms <- data.frame(a=rep(1,nrow(data)))
      names(linterms) <- "(Intercept)"
      attr(linterms, "names") <- names(linterms)
      return(
        list(linterms = linterms,
             smoothterms = NULL,
             deepterms = NULL)
      )
    }else{ return(NULL) }
  }
  trmstrings <- attr(tf, "term.labels")
  # check for missing covariates in data
  for(j in trmstrings)
  {
    if(!grepl("\\(",j) | !grepl("\\)",j)){
      if(xor(!grepl("\\(",j),  !grepl("\\)",j))){
        stop("Terms in formula with only one parantheses.")
      }else{
        # make pseudo parantheses so regmatch detects variable
        # in the following lines
        j <- paste0("(",j,")")
      }
    }
    vars <- extract_from_special(j)
    # drop terms that specify a s-term specification
    vars <- vars[!grepl("=", vars, fixed=T)]
    whatsleft <- setdiff(vars, variable_names)
    if(length(whatsleft) > 0){
      stop(paste0("data for ", paste(whatsleft, collapse = ","), " in ",
                  j, " not found"))
    }
  }
  #
  terms <- sapply(trmstrings, function(trm) as.call(parse(text=trm))[[1]],
                  simplify=FALSE)
  # get formula environment
  # frmlenv <- environment(formula)
  # get linear terms
  desel <- unlist(attr(tf, "specials"))
  if(!is.null(desel))
    linterms <- data[,attr(tf, "term.labels")[-1*desel], drop=FALSE] else
                       linterms <- data[,attr(tf, "term.labels"), drop=FALSE]
  if(intercept & attr(tf,"intercept"))
    linterms <- cbind("(Intercept)" = rep(1,nrow(linterms)), linterms)
  attr(linterms, "names") <- names(linterms)
  # get deep terms
  dterms <- trmstrings[grepl("d\\(.*\\)", trmstrings)]
  if(length(dterms)==0) deepterms <- NULL else
    deepterms <- data[,extract_from_special(dterms),drop=FALSE]
  attr(deepterms, "names") <- names(deepterms)
  # get gam terms
  spec <- attr(tf, "specials")
  sTerms <- terms[unlist(spec[names(spec)!="d"])]
  if(length(sTerms)>0)
  {
    smoothterms <- sapply(sTerms,
                          function(t) smoothCon(eval(t), data=data, knots=NULL))
    # ranks <- sapply(smoothterms, function(x) rankMatrix(x$X, method = 'qr', warn.t = FALSE))
    if(is.null(df)) df <- min(sapply(smoothterms, "[[", "df"))
    if(is.null(defaultSmoothing))
      defaultSmoothing = function(st){
        # TODO: Extend for TPs (S[[1]] is only the first matrix)
        st$sp = DRO(st$X, df = df, dmat = st$S[[1]])["lambda"]
        return(st)
      }
    smoothterms[sapply(smoothterms,function(x) is.null(x$sp))] <-
      lapply(smoothterms[sapply(smoothterms,function(x) is.null(x$sp))], defaultSmoothing)
    attr(smoothterms, "names") <- sapply(names(smoothterms),
                                         function(x){
                                           vars <- extract_from_special(x)
                                           vars <- vars[!grepl("=", vars, fixed=T)]
                                           paste(vars, collapse=",")
                                           })
  # values in smooth construct list have the following items
  # (see also ?mgcv::smooth.construct)
  #
  # X: model matrix
  # S: psd penalty matrix
  # rank: array with ranks of penalties
  # null.space.dim: dimension of penalty null space
  # C: identifiability constraints on term (per default sum-to-zero constraint)
  # and potential further entries
  }else{
    smoothterms <- NULL
  }

  return(list(linterms = linterms,
              smoothterms = smoothterms,
              deepterms = deepterms))
}

make_cov <- function(pcf, newdata=NULL,
                     convertfun = function(x)
                       tf$constant(x, dtype="float32")){

  if(is.null(newdata))
    input_cov <- lapply(pcf, function(x){
      if(is.null(x$deepterms)) return(NULL) else
        return(as.matrix(x$deepterms))
      }) else
      input_cov <- lapply(pcf, function(x){
        if(length(intersect(names(x$deepterms),names(newdata)))>0)
          return(newdata[,names(x$deepterms),drop=FALSE]) else
            return(NULL)
      })
  input_cov <- c(input_cov,
                 lapply(pcf, function(x){
                   ret <- NULL
                   if(!is.null(x$linterms))
                     if(is.null(newdata))
                       ret <- x$linterms else{
                         if("(Intercept)" %in% names(x$linterms))
                           newdata[,"(Intercept)"] <- rep(1,nrow(newdata))
                         ret <- newdata[,names(x$linterms),drop=FALSE]
                       }
                       if(!is.null(x$smoothterms))
                       {
                         if(!is.null(newdata)){
                           Xp <- lapply(x$smoothterms, function(sm)
                             PredictMat(sm,newdata))
                         }else{
                           Xp <- lapply(x$smoothterms, "[[", "X")
                         }
                         st <- do.call("cbind", Xp)
                         ret <- cbind(ret, st)
                         ret <- array(as.matrix(ret), dim = c(nrow(ret),1,ncol(ret)))
                       }
                       return(ret)
                 }))

  # just use the ones with are actually modeled
  input_cov <- input_cov[!sapply(input_cov, function(x) is.null(x) | 
                                   (length(x)==1 && is.null(x[[1]])) | 
                                   NCOL(x)==0)]
  input_cov_isdf <- sapply(input_cov, is.data.frame)
  input_cov[which(input_cov_isdf)] <- lapply(input_cov[which(input_cov_isdf)],
                                             as.matrix)
  input_cov <- lapply(input_cov, convertfun)
  return(input_cov)

}

get_names <- function(x)
{

  lret <- list(linterms = NULL,
               smoothterms = NULL,
               deepterms = NULL)
  if(!is.null(x$linterms)) lret$linterms <- names(x$linterms)
  if(!is.null(x$smoothterms)) lret$smoothterms <- sapply(x$smoothterms,"[[","label")
  if(!is.null(x$deepterms)) lret$deepterms <- names(x$deepterms)
  return(lret)
}

get_indices <- function(x)
{
  if(!is.null(x$linterms) & 
     !(length(x$linterms)==1 & is.null(x$linterms[[1]]))) 
    ncollin <- ncol(x$linterms) else ncollin <- 0
  if(!is.null(x$smoothterms))
    bsdims <- sapply(x$smoothterms, function(y){
      if(is.null(y$margin)) return(y$bs.dim) else
        # Tensorprod
        return(prod(sapply(y$margin,"[[", "bs.dim")))
      }) else bsdims <- c()
    ind <- if(ncollin > 0) seq(1, ncollin, by = 1) else c()
    end <- if(ncollin > 0) ind else c()
    if(length(bsdims) > 0) ind <- c(ind, max(c(ind,0))+1, max(c(ind+1,1)) +
                                      cumsum(bsdims[-length(bsdims)]))
    if(length(bsdims) > 0) end <- c(end, max(c(end,0)) +
                                      cumsum(bsdims))

    return(data.frame(start=ind, end=end, type=c(rep("lin",ncollin),
                                                 rep("smooth",length(bsdims))))
    )
}

prepare_newdata <- function(pfc, data, pred = TRUE, index = NULL)
{
  n_obs <- nrow(data)
  input_cov_new <- make_cov(pfc, data)
  ox <- lapply(pfc, make_orthog)
  if(!is.null(index)){
    ox <- lapply(ox, function(xox) xox[index,,drop=FALSE])
  }
  newdata_processed <- append(
    c(unname(input_cov_new),
      list(c(1-pred,rep(0,n_obs-1)))),
    unname(ox[!sapply(ox, is.null)]))
  return(newdata_processed)
}

coefkeras <- function(model)
{
  
  layer_names <- sapply(model$layers, "[[", "name")
  layers_names_structured <- layer_names[
    grep("structured_", layer_names)
  ]
  unlist(sapply(layers_names_structured, 
                function(name) model$get_layer(name)$get_weights()[[1]]))
}

make_cv_list_simple <- function(data_size, folds, seed = 42, shuffle = TRUE)
{
  
  set.seed(seed)
  suppressWarnings(
    mysplit <- split(sample(1:data_size), f = rep(1:folds, each = data_size/folds))
  )
  lapply(mysplit, function(test_ind) list(train_ind = setdiff(1:data_size, test_ind),
                                          test_ind = test_ind))
  
}

extract_cv_result <- function(res){
  
  losses <- sapply(res, "[[", "metrics")
  trainloss <- data.frame(losses[1,])
  validloss <- data.frame(losses[2,])
  weightshist <- lapply(res, "[[", "weighthistory")
  
  return(list(trainloss=trainloss,validloss=validloss,weight=weightshist))
  
}

#' Plot CV results from deepregression
#'
#' @method plot drCV
#' @param x \code{drCV} object returned by \code{cv.deepregression}
#' @param what character indicating what to plot (currently supported 'loss'
#' or 'weights')
#' 
#' @export
#' 
plot.drCV <- function(x, what=c("loss","weight"), ...){
  
  
  cres <- extract_cv_result(x)
  
  what <- match.arg(what)
  
  if(what=="loss"){
    
    loss <- cres$trainloss
    mean_loss <- apply(loss, 1, mean)
    vloss <- cres$validloss
    mean_vloss <- apply(vloss, 1, mean)
    
    par(mfrow=c(1,2))
    matplot(loss, type="l", col="black", ..., ylab="loss", xlab="epoch")
    points(1:(nrow(loss)), mean_loss, type="l", col="red", lwd=2)
    abline(v=which.min(mean_loss), lty=2)
    matplot(vloss, type="l", col="black", ..., ylab="validation loss", xlab="epoch")
    points(1:(nrow(vloss)), mean_vloss, type="l", col="red", lwd=2)
    abline(v=which.min(mean_vloss), lty=2)
    
  }else{
    
      
    
  }

}


stop_iter_cv_result <- function(res, FUN = mean, 
                                loss = "validloss",
                                whichFUN = which.min)
{
  
  whichFUN(apply(extract_cv_result(res)[[loss]], 1, FUN))
  
}

#' Generate folds for CV out of one hot encoded matrix
#' 
#' @param mat matrix with columns corresponding to folds
#' and entries corresponding to a one hot encoding
#' @param val_train the value corresponding to train, per default 0
#' @param val_test the value corresponding to test, per default 1
#' 
#' @details 
#' \code{val_train} and \code{val_test} can both be a set of value
#' 
#' @export
make_folds <- function(mat, val_train=0, val_test=1)
{
  
  apply(mat, 2, function(x){
    list(train = which(x %in% val_train),
         test = which(x %in% val_test))
  })
  
}