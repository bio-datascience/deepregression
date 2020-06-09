# function that extracts variables from special symbols in formulae
extract_from_special <- function(x)
  trimws(
    strsplit(regmatches(x,
                        gregexpr("(?<=\\().*?(?=\\))", x, perl=T))[[1]],
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
                  dense_shape = as.integer(X@Dim))

}

NCOL0 <- function(x)
{
  if(is.null(x))
    return(0)
  return(NCOL(x))
}

#### from mgcv
uniquecombs <- function(x,ordered=FALSE) {
  ## takes matrix x and counts up unique rows
  ## `unique' now does this in R
  if (is.null(x)) stop("x is null")
  if (is.null(nrow(x))||is.null(ncol(x))) x <- data.frame(x)
  recheck <- FALSE
  if (inherits(x,"data.frame")) {
    xoo <- xo <- x
    ## reset character, logical and factor to numeric, to guarantee that text versions of labels
    ## are unique iff rows are unique (otherwise labels containing "*" could in principle
    ## fool it).
    is.char <- rep(FALSE,length(x))
    for (i in 1:length(x)) {
      if (is.character(xo[[i]])) {
        is.char[i] <- TRUE
        xo[[i]] <- as.factor(xo[[i]])
      }
      if (is.factor(xo[[i]])||is.logical(xo[[i]])) x[[i]] <- as.numeric(xo[[i]])
      if (!is.numeric(x[[i]])) recheck <- TRUE ## input contains unknown type cols
    }
    #x <- data.matrix(xo) ## ensure all data are numeric
  } else xo <- NULL
  if (ncol(x)==1) { ## faster to use R
    xu <- if (ordered) sort(unique(x[,1])) else unique(x[,1])
    ind <- match(x[,1],xu)
    if (is.null(xo)) x <- matrix(xu,ncol=1,nrow=length(xu)) else {
      x <-  data.frame(xu)
      names(x) <- names(xo)
    }
  } else { ## no R equivalent that directly yields indices
    if (ordered) {
      chloc <- Sys.getlocale("LC_CTYPE")
      Sys.setlocale("LC_CTYPE","C")
    }
    ## txt <- paste("paste0(",paste("x[,",1:ncol(x),"]",sep="",collapse=","),")",sep="")
    ## ... this can produce duplicate labels e.g. x[,1] = c(1,11), x[,2] = c(12,2)...
    ## solution is to insert separator not present in representation of a number (any
    ## factor codes are already converted to numeric by data.matrix call above.)
    txt <- paste("paste0(",paste("x[,",1:ncol(x),"]",sep="",collapse=",\"*\","),")",sep="")
    xt <- eval(parse(text=txt)) ## text representation of rows
    dup <- duplicated(xt)       ## identify duplicates
    xtu <- xt[!dup]             ## unique text rows
    x <- x[!dup,]               ## unique rows in original format
    #ordered <- FALSE
    if (ordered) { ## return unique in same order regardless of entry order
      ## ordering of character based labels is locale dependent
      ## so that e.g. running the same code interactively and via
      ## R CMD check can give different answers.
      coloc <- Sys.getlocale("LC_COLLATE")
      Sys.setlocale("LC_COLLATE","C")
      ii <- order(xtu)
      Sys.setlocale("LC_COLLATE",coloc)
      Sys.setlocale("LC_CTYPE",chloc)
      xtu <- xtu[ii]
      x <- x[ii,]
    }
    ind <- match(xt,xtu)   ## index each row to the unique duplicate deleted set

  }
  if (!is.null(xo)) { ## original was a data.frame
    x <- as.data.frame(x)
    names(x) <- names(xo)
    for (i in 1:ncol(xo)) {
      if (is.factor(xo[,i])) { ## may need to reset factors to factors
        xoi <- levels(xo[,i])
        x[,i] <- if (is.ordered(xo[,i])) ordered(x[,i],levels=1:length(xoi),labels=xoi) else
          factor(x[,i],levels=1:length(xoi),labels=xoi)
        contrasts(x[,i]) <- contrasts(xo[,i])
      }
      if (is.char[i]) x[,i] <- as.character(x[,i])
      if (is.logical(xo[,i])) x[,i] <- as.logical(x[,i])
    }
  }
  if (recheck) {
    if (all.equal(xoo,x[ind,],check.attributes=FALSE)!=TRUE)
      warning("uniquecombs has not worked properly")
  }
  attr(x,"index") <- ind
  x
} ## uniquecombs

### from mgcv
compress_data <- function(dat, m = NULL)
{
  d <- ncol(dat) ## number of variables to deal with
  n <- nrow(dat) ## number of data/cases
  if (is.null(m)) m <- if (d==1) 1000 else if (d==2) 100 else 25 else
    if (d>1) m <- round(m^{1/d}) + 1

  mf <- mm <- 1 ## total grid points for factor and metric
  for (i in 1:d) if (is.factor(dat[,i])) {
    mf <- mf * length(unique(as.vector(dat[,i])))
  } else {
    mm <- mm * m
  }
  if (is.matrix(dat[[1]])) { ## must replace matrix terms with vec(dat[[i]])
    dat0 <- data.frame(as.vector(dat[[1]]))
    if (d>1) for (i in 2:d) dat0[[i]] <- as.vector(dat[[i]])
    names(dat0) <- names(dat)
    dat <- dat0;rm(dat0)
  }
  xu <- uniquecombs(dat,TRUE)
  if (nrow(xu)>mm*mf) { ## too many unique rows to use only unique
    for (i in 1:d) if (!is.factor(dat[,i])) { ## round the metric variables
      xl <- range(dat[,i])
      xu <- seq(xl[1],xl[2],length=m)
      dx <- xu[2]-xu[1]
      kx <- round((dat[,i]-xl[1])/dx)+1
      dat[,i] <- xu[kx] ## rounding the metric variables
    }
    xu <- uniquecombs(dat,TRUE)
  }
  k <- attr(xu,"index")
  ## shuffle rows in order to avoid induced dependencies between discretized
  ## covariates (which can mess up gam.side)...
  ## Any RNG setting should be done in routine calling this one!!

  ii <- sample(1:nrow(xu),nrow(xu),replace=FALSE) ## shuffling index

  xu[ii,] <- xu  ## shuffle rows of xu
  k <- ii[k]     ## correct k index accordingly
  ## ... finished shuffle
  ## if arguments were matrices, then return matrix index
  if (length(k)>n) k <- matrix(k,nrow=n)
  k -> attr(xu,"index")
  xu
}

# get contents from formula
get_contents <- function(lf, data, df,
                         variable_names,
                         network_names,
                         intercept = TRUE,
                         defaultSmoothing,
                         absorb_cons = TRUE,
                         null_space_penalty = FALSE){
  # extract which parts are modelled as deep parts
  # which by smooths, which linear
  specials <- c("s", "te", "ti", network_names)
  tf <- terms.formula(lf, specials=specials)
  
  if(length(attr(tf, "term.labels"))==0){
    if(intercept & attr(tf,"intercept")){
      if(is.data.frame(data)) linterms <- data.frame(a=rep(1,nrow(data))) else
        linterms <- data.frame(a=rep(1,nROW(data)))
      names(linterms) <- "(Intercept)"
      attr(linterms, "names") <- names(linterms)
      ret <- list(linterms = linterms,
                  smoothterms = NULL,
                  deepterms = NULL)
      attributes(ret) <-
        c(attributes(ret),
          list(formula = lf,
               df = df,
               variable_names = variable_names,
               network_names = network_names,
               intercept = intercept,
               defaultSmoothing = defaultSmoothing)
        )
      return(ret)
    }else{ return(NULL) }
  }
  trmstrings <- attr(tf, "term.labels")
  # if(length(setdiff(c(gsub("(.*)\\(.*\\)","\\1",trmstrings),
  #                     variable_names),
  #                   specials))>0)
  #   stop("It seems that you are using non-valid terms in the formula ",
  #        "or specified a list_of_deep_models without names.")
  
  # check for weird line break behaviour produced by terms.formula
  trmstrings <- unname(sapply(trmstrings, function(x)
    gsub("\\\n\\s+", "", x, fixed=F)))
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
    # replace . in formula
    if(length(vars)==1 && vars==".")
    {
      ff <- as.character(lf)[[2]]
      net_w_dot <- sapply(network_names, function(x) grepl(paste0(x,"\\("),j))
      if(grepl("d\\(",j) | any(net_w_dot))
        ff <- gsub("\\.", paste(variable_names, collapse=","), ff) else
          ff <- gsub(".", paste(variable_names, collapse="+"), ff)
        return(get_contents(lf = as.formula(paste0("~ ", ff)),
                            data = data,
                            df = df,
                            variable_names = variable_names,
                            intercept = intercept,
                            network_names = network_names,
                            defaultSmoothing = defaultSmoothing))
    }
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
  # if(is.data.frame(data)){
  #   if(!is.null(desel)) linterms <-
  #       data[,attr(tf, "term.labels")[-1*desel], drop=FALSE] else
  #       linterms <- data[,attr(tf, "term.labels"), drop=FALSE]
  # }else{
  if(!is.null(desel)){
    ind <- attr(tf, "term.labels")[-1*desel]
    if(length(ind)!=0) linterms <- as.data.frame(data[ind]) else
      linterms <- data.frame(dummy=1:nROW(data))[character(0)]
  }else{
    # else
    #     stop("When using only structured terms, data must be a data.frame")
    if(length(attr(tf,"term.labels"))>0)
      linterms <- as.data.frame(data[attr(tf, "term.labels")]) else
        linterms <- data.frame(dummy=1:nROW(data))[character(0)]
      # }
  }
  if(intercept & attr(tf,"intercept"))#{
    # if(NCOL(linterms)==0)
    if(NROW(linterms)==0)
      linterms <- data.frame("(Intercept)" = rep(1,nROW(data))) else
        linterms <- cbind("(Intercept)" = rep(1,nROW(data)),
                          as.data.frame(linterms))# else

  attr(linterms, "names") <- names(linterms)

  # get gam terms
  spec <- attr(tf, "specials")
  sTerms <- terms[unlist(spec[names(spec) %in% c("s", "te", "ti")])]
  if(any(!sapply(spec[c("te","ti")], is.null)))
    warning("2-dimensional smooths and higher currently not well tested.")
  if(length(sTerms)>0)
  {
    names_sTerms <- names(sTerms)
    terms_w_s <- lapply(names(sTerms), extract_from_special)
    by_vars <- lapply(terms_w_s, function(x) sapply(x, function(y){ 
      if(grepl("by.*\\=",y)) return(trimws(gsub("by.*\\=(.*)","\\1",y))) else return(y)}))
    terms_w_s <- lapply(terms_w_s, function(x) sapply(x, function(y){ 
      if(grepl("by.*\\=",y)) return(trimws(gsub("by.*\\=(.*)","\\1",y))) else return(y)}))
    terms_w_s <- lapply(terms_w_s, function(x) x[!grepl("=", x, fixed=T)])
    smoothterms <- 
      lapply(sTerms,
             function(t)
               smoothCon(eval(t),
                         data=data.frame(data[unname(unlist(terms_w_s))]),
                         knots=NULL, absorb.cons = absorb_cons,
                         null.space.penalty = null_space_penalty))
    
    # ranks <- sapply(smoothterms, function(x) rankMatrix(x$X, method = 'qr',
    # warn.t = FALSE))
    if(is.null(df)) df <- pmax(min(sapply(smoothterms, function(x) x[[1]]$df)) - 
                                 null_space_penalty, 1)
    if(is.null(defaultSmoothing))
      defaultSmoothing = function(st){
        # TODO: Extend for TPs (S[[1]] is only the first matrix)
        if(length(st[[1]]$S)==1) S <- st[[1]]$S[[1]] else
          S <- Reduce("+", st[[1]]$S)
        st[[1]]$sp = DRO(st[[1]]$X, df = df, dmat = S)["lambda"] + null_space_penalty
        return(st)
      }
    smoothterms[sapply(smoothterms,function(x) is.null(x[[1]]$sp))] <-
      lapply(smoothterms[sapply(smoothterms,function(x) is.null(x[[1]]$sp))],
             defaultSmoothing)
    attr(smoothterms, "names") <-
      unlist(lapply(names_sTerms,
             function(x){
               vars <- extract_from_special(x)
               vars <- vars[!grepl("=", vars, fixed=T) | grepl("by.*\\=",vars)]
               # rep <- FALSE
               # if(any(grepl("by.*\\=",vars))){
               #   fac <- trimws(gsub("by.*\\=(.*)","\\1",vars[grepl("by.*\\=",vars)]))
               #   rep <- TRUE
               #   }
               # vars[grepl("by.*\\=",vars)] <- gsub("(\\s+)\\=(\\s+)","_",vars[grepl("by.*\\=",vars)])
               #ret <- 
                 paste(vars, collapse=",")
               # if(rep) paste0(ret, 1:nlevels(data[[fac]])) else ret
             }))
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

  # get deep terms
  dterms <- sapply(paste0(network_names,"\\("), function(x) trmstrings[grepl(x,trmstrings)])
  if(all(sapply(dterms,length)==0)){
    deepterms <- NULL
  }else{
    deepterms <- lapply(dterms[sapply(dterms,length)>0], function(dt){
      if(is.data.frame(data)){
        deepterms <- data[,extract_from_special(dt),drop=FALSE]
        attr(deepterms, "names") <- names(deepterms)
      }else{
        deepterms <- data[extract_from_special(dt)]

        if(length(extract_from_special(dt))>1)
          deepterms <- as.data.frame(deepterms)

      }
      attr(deepterms, "names") <- names(deepterms)
      return(deepterms)
    })
    if(length(network_names)==1)
      names(deepterms) <- rep(network_names, length(deepterms)) else
        names(deepterms) <- network_names[sapply(dterms,length)>0]
  }

  ret <- list(linterms = linterms,
              smoothterms = smoothterms,
              deepterms = deepterms)

  attributes(ret) <-
    c(attributes(ret),
    list(formula = lf,
         df = df,
         variable_names = variable_names,
         network_names = network_names,
         intercept = intercept,
         defaultSmoothing = defaultSmoothing)
    )

  return(ret)
}

get_contents_newdata <- function(pcf, newdata)
  lapply(pcf, function(x) get_contents(lf = attr(x, "formula"),
                                       data = newdata,
                                       df = attr(x, "df"),
                                       variable_names = attr(x, "variable_names"),
                                       network_names = attr(x, "network_names"),
                                       intercept = attr(x, "intercept"),
                                       defaultSmoothing = attr(x, "defaultSmoothing")))

make_cov <- function(pcf, newdata=NULL,
                     convertfun = function(x)
                       tf$constant(x, dtype="float32"),
                     pred = !is.null(newdata)){

  if(is.null(newdata)){
    input_cov <- lapply(pcf, function(x){
      if(is.null(x$deepterms)) return(NULL) else
        return(x$deepterms)
    })
  }else{
    input_cov <- lapply(pcf, function(x){
      if(length(intersect(sapply(x$deepterms,
                                 function(y) names(y)),names(newdata)))>0){
        ret <- lapply(x$deepterms, function(y){
          if(is.data.frame(y)){
            return(as.data.frame(newdata[names(y)]))
          }else{
            return(newdata[names(y)])
          }
        })

      }else if(is.list(x$deepterms) & all(sapply(x$deepterms, class)=="data.frame")){

        return(lapply(x$deepterms, function(y) data.frame(newdata[names(y)])))

      }else{ return(NULL) }
    })
  }
  if(is.list(input_cov) & all(sapply(input_cov, is.list)))
    input_cov <- unlist(input_cov, recursive = F, use.names = F)
  input_cov_isdf <- sapply(input_cov, is.data.frame)
  if(sum(input_cov_isdf)>0)
    input_cov[which(input_cov_isdf)] <-
    lapply(input_cov[which(input_cov_isdf)], as.matrix)

  if(!is.null(newdata) & pred)
    pcfnew <- get_contents_newdata(pcf, newdata)

  input_cov <- c(input_cov,
                 lapply(1:length(pcf), function(i){
                   x = pcf[[i]]
                   ret <- NULL
                   if(!is.null(x$linterms))
                     if(is.null(newdata)){
                       if(any(sapply(x$linterms,is.factor))){
                         ret <- model.matrix(~ 1 + ., data = x$linterms)[,-1]
                       }else{
                         ret <- model.matrix(~ 0 + ., data = x$linterms)
                       }
                     }else{
                       if("(Intercept)" %in% names(x$linterms))
                         newdata$`(Intercept)` <- rep(1, nROW(newdata))
                       if(any(sapply(x$linterms,is.factor))){
                         ret <- model.matrix(~ 1 + ., data = newdata[names(x$linterms)])[,-1]
                       }else{
                         ret <- model.matrix(~ 0 + ., data = newdata[names(x$linterms)])
                       }
                     }
                     if(!is.null(x$smoothterms))
                     {
                       # put all terms that 
                       # names_sTerms <- names(x$smoothterms)
                       # byfac <- intersect(names(newdata),
                       #                    unique(gsub(".*,by_(.*)([0-9]+)$","\\1", names_sTerms)))
                       # if(length(byfac)>0){
                       #   tog_list <- vector("list", length(byfac))
                       #   for(j in 1:length(byfac)){
                       #     bf <- byfac[j]
                       #     byf <- paste0(",by_", bf)
                       #     these_s <- grepl(byf, names(x$smoothterms))
                       #     var_before <- unique(gsub(paste0("(.*)",byf,".*"),"\\1",
                       #                               names(x$smoothterms)))
                       #     tog_list[[j]] <- vector("list", length(var_before))
                       #     for(k in 1:length(var_before)){
                       #       vb <- var_before[k]
                       #       these_s_v <- grepl(paste0(vb,",by_",bf), names(x$smoothterms))
                       #       tog_list[[j]][[k]] <- which(these_s & these_s_v)
                       #     }
                       #   }
                       #   tog_list <- unlist(tog_list, recursive = F)
                       #   xsm <- x$smoothterms[!these_s]
                       #   for(tl in tog_list){
                       #     xsm <- c(xsm, list(x$smoothterms[tl]))
                       #   }
                       # }
                       if(!is.null(newdata) & !pred){
                         Xp <- lapply(x$smoothterms, function(sm)
                         {
                           if(length(sm)==1){ 
                             sm <- sm[[1]]
                             sterms <- sm$term
                             PredictMat(sm,as.data.frame(newdata[sterms]))
                           }else{
                             sterms <- c(sm[[1]]$term, sm[[1]]$by)
                             do.call("cbind", lapply(sm, function(smm)
                               PredictMat(smm,as.data.frame(newdata[sterms]))))
                           }

                         })
                       }else if(!is.null(newdata) & pred){
                         Xp <- lapply(pcfnew[[i]]$smoothterms, function(x) 
                           do.call("cbind", lapply(x, "[[", "X")))
                       }else{
                         Xp <- lapply(x$smoothterms, function(x)  
                           do.call("cbind", lapply(x, "[[", "X")))
                       }
                       st <- do.call("cbind", Xp)
                       if(!is.null(ret)){
                         ret <- cbind(as.data.frame(ret), st)

                       }else{
                         ret <- st
                       }
                       ret <- array(as.matrix(ret),
                                    dim = c(nrow(ret),ncol(ret)))
                     }
                     return(ret)
                 }))

  # just use the ones with are actually modeled
  input_cov <- input_cov[!sapply(input_cov, function(x) is.null(x) |
                                   (length(x)==1 && is.null(x[[1]])) |
                                   NCOL(x)==0)]
  input_cov <- unlist_order_preserving(input_cov)
  list_len_1 <- sapply(input_cov, function(x) is.list(x) & length(x)==1)
  input_cov[list_len_1] <- lapply(input_cov[list_len_1], function(x) x[[1]])
  input_cov[sapply(lapply(input_cov,dim),is.null)] <-
    lapply(input_cov[sapply(lapply(input_cov,dim),is.null)],
           function(x) matrix(x, ncol=1))
  input_cov_isdf <- sapply(input_cov, is.data.frame)
  if(sum(input_cov_isdf)>0)
    input_cov[which(input_cov_isdf)] <-
    lapply(input_cov[which(input_cov_isdf)], as.matrix)
  input_cov[!sapply(input_cov,is.factor)] <-
    lapply(input_cov[!sapply(input_cov,is.factor)], convertfun)
  return(input_cov)

}

get_names <- function(x)
{

  lret <- list(linterms = NULL,
               smoothterms = NULL,
               deepterms = NULL)
  if(!is.null(x$linterms)) lret$linterms <- names(x$linterms)
  if(!is.null(x$smoothterms)) lret$smoothterms <-
      sapply(x$smoothterms,function(x)x[[1]]$label)
  if(!is.null(x$deepterms)) lret$deepterms <- names(x$deepterms)
  return(lret)
}

get_indices <- function(x)
{
  if(!is.null(x$linterms) &
     !(length(x$linterms)==1 & is.null(x$linterms[[1]])))
    ncollin <- ncol(x$linterms) else ncollin <- 0
    if(!is.null(x$smoothterms))
      bsdims <- unlist(lapply(x$smoothterms, function(y){
        if(is.null(y[[1]]$margin) & y[[1]]$by=="NA") 
          return(y[[1]]$bs.dim-attr(y[[1]],"nCons")) else if(
            is.null(y[[1]]$margin) & y[[1]]$by!="NA")
            return(sapply(y, "[[", "bs.dim")-attr(y[[1]],"nCons")) else
              # Tensorprod
              return(prod(sapply(y[[1]]$margin,"[[", "bs.dim")))
      })) else bsdims <- c()
      ind <- if(ncollin > 0) seq(1, ncollin, by = 1) else c()
      end <- if(ncollin > 0) ind else c()
      if(length(bsdims) > 0) ind <- c(ind, max(c(ind,0))+1, max(c(ind+1,1)) +
                                        cumsum(bsdims[-length(bsdims)]))
      if(length(bsdims) > 0) end <- c(end, max(c(end,0)) +
                                        cumsum(bsdims))

      return(data.frame(start=ind, end=end,
                        type=c(rep("lin",ncollin),
                               rep("smooth",length(bsdims))))
      )
}

prepare_newdata <- function(pfc, data, pred = FALSE, index = NULL)
{
  n_obs <- nROW(data)
  input_cov_new <- make_cov(pfc, data, pred = FALSE)
  if(pred & !is.null(data))
    pfc <- get_contents_newdata(pfc, data)
  ox <- lapply(pfc, make_orthog)
  if(pred){
    ox <- unlist(lapply(ox, function(x_per_param)
      if(is.null(x_per_param)) return(NULL) else
        unlist(lapply(x_per_param[!sapply(x_per_param,is.null)], function(x)
          tf$constant(x*0, dtype="float32")))), recursive=F)
  }
  if(!is.null(index)){
    ox <- unlist(lapply(ox, function(x_per_param)
      if(is.null(x_per_param)) return(NULL) else
        unlist(lapply(x_per_param[!sapply(x_per_param,is.null)], function(xox)
          tf$constant(as.matrix(xox)[index,,drop=FALSE],
                      dtype="float32")))),
      recursive=F)
  }
  if(is.null(index) & !pred){
    ox <- unlist(lapply(ox, function(x_per_param)
      if(is.null(x_per_param)) return(NULL) else
        unlist(lapply(x_per_param[!sapply(x_per_param,is.null)], function(x)
          tf$constant(x, dtype="float32")))), recursive=F)
  }
  newdata_processed <- append(
    c(unname(input_cov_new)),
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
    mysplit <- split(sample(1:data_size),
                     f = rep(1:folds, each = data_size/folds))
  )
  lapply(mysplit, function(test_ind)
    list(train_ind = setdiff(1:data_size, test_ind),
         test_ind = test_ind))

}

extract_cv_result <- function(res, name_loss = "loss", name_val_loss = "val_loss"){

  losses <- sapply(res, "[[", "metrics")
  trainloss <- data.frame(losses[name_loss,])
  validloss <- data.frame(losses[name_val_loss,])
  weightshist <- lapply(res, "[[", "weighthistory")

  return(list(trainloss=trainloss,
              validloss=validloss,
              weight=weightshist))

}

#' Plot CV results from deepregression
#'
#' @method plot drCV
#' @param x \code{drCV} object returned by \code{cv.deepregression}
#' @param what character indicating what to plot (currently supported 'loss'
#' or 'weights')
#' @param ... further arguments passed to \code{matplot}
#'
#' @export
#'
plot.drCV <- function(x, what=c("loss","weight"), ...){

  .pardefault <- par()
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
    matplot(vloss, type="l", col="black", ...,
            ylab="validation loss", xlab="epoch")
    points(1:(nrow(vloss)), mean_vloss, type="l", col="red", lwd=2)
    abline(v=which.min(mean_vloss), lty=2)
    suppressWarnings(par(.pardefault))

  }else{

    stop("Not implemented yet.")

  }

  invisible(NULL)

}


stop_iter_cv_result <- function(res, thisFUN = mean,
                                loss = "validloss",
                                whichFUN = which.min)
{

  whichFUN(apply(extract_cv_result(res)[[loss]], 1, FUN=thisFUN))

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

subset_array <- function(x, index)
{

  dimx <- dim(x)
  if(is.null(dimx)) dimx = 1
  eval(parse(text=paste0("x[index",
                         paste(rep(",", length(dimx)-1),collapse=""),
                         ",drop=FALSE]")))

}

# nrow for list
nROW <- function(x)
{
  NROW(x[[1]])
}

nCOL <- function(x)
{
  lapply(x, function(y) if(is.null(dim(y))) 1 else dim(y)[-1])
}

nestNCOL <- function(x)
{
  
  res <- list()
  for(i in 1:length(x)){
   
    if(is.list(x[[i]]) & length(x[[i]])>=1 & !is.null(x[[i]][[1]])){
      res[[i]] <- nestNCOL(x[[i]])
    }else if((is.list(x[[i]]) & length(x[[i]])==0) | is.null(x[[i]][[1]])){
      res[[i]] <- 0
    }else{
      res[[i]] <- NCOL(x[[i]]) 
    }
     
  }
  
  return(res)
}

ncol_lint <- function(z)
{

  if(is.null(z)) return(0)
  z_num <- NCOL(z[,!sapply(z,is.factor),drop=F])
  facs <- sapply(z,is.factor)
  if(length(facs)>0) z_fac <- sapply(z[,facs,drop=F], nlevels) else
    z_fac <- 0
  if(length(z_fac)==0) z_fac <- 0 else z_fac <- z_fac-1
  return(sum(c(z_num, z_fac)))

}

unlist_order_preserving <- function(x)
{

  x_islist <- sapply(x, is.list)
  if(any(x_islist)){

    for(w in which(x_islist)){

      beginning <- if(w>1) x[1:(w-1)] else list()
      end <- if(w<length(x))
        x[(w+1):length(x)] else list()

      is_data_frame <- is.data.frame(x[[w]])
      if(is_data_frame) dfxw <- as.matrix(x[[w]])
      len_bigger_one <- !is_data_frame & length(x[[w]])>1 & is.list(x[[w]])
      if(is_data_frame) x <- append(beginning, list(dfxw)) else
        x <- append(beginning, x[[w]])
      x <- append(x, end)
      if(len_bigger_one) return(unlist_order_preserving(x))

    }

  }

  return(x)

}

get_family_name <- function(dist) gsub(".*(^|/)(.*)/$", "\\2", dist$name)
