WeightHistory <- R6::R6Class("WeightHistory",
                             inherit = KerasCallback,
                             
                             public = list(
                               
                               weights_last_layer = NULL,
                               
                               on_epoch_end = function(batch, logs = list()) {
                                 self$weights_last_layer <- 
                                   cbind(self$weights_last_layer, 
                                         coefkeras(self$model))
                               }
                             ))


# overwrite keras:::KerasMetricsCallback class
KerasMetricsCallback_custom <- R6::R6Class("KerasMetricsCallback",
                               
                               inherit = KerasCallback,
                               
                               public = list(
                                 
                                 # instance data
                                 metrics = list(),
                                 metrics_viewer = NULL,
                                 weights_last_layer = NULL,
                                 view_metrics = FALSE,
                                 
                                 initialize = function(view_metrics = FALSE) {
                                   self$view_metrics <- view_metrics
                                 },
                                 
                                 on_train_begin = function(logs = NULL) {
                                   
                                   # add weight_diff to metrics
                                   self$params$metrics <- c(self$params$metrics, 
                                                            "weight_diff")
                                   # strip validation metrics if do_validation is FALSE (for
                                   # fit_generator and fitting TF record the val_ metrics are
                                   # passed even though no data will be provided for them)
                                   if (!self$params$do_validation) {
                                     self$params$metrics <- Filter(function(metric) {
                                       !grepl("^val_", metric)
                                     }, self$params$metrics)
                                   }
                                   
                                   # initialize metrics & weights
                                   for (metric in self$params$metrics)
                                     self$metrics[[metric]] <- numeric()
                                   
                                   # handle metrics
                                   if (length(logs) > 0)
                                   {
                                     self$on_metrics(logs, 0.5)
                                   }
                                   
                                   if (tfruns::is_run_active()) {
                                     self$write_params(self$params)
                                     self$write_model_info(self$model)
                                   }
                                 },
                                 
                                 on_epoch_end = function(epoch, logs = NULL) {
                                   
                                   # handle metrics
                                   if(epoch==0){
                                     self$weights_last_layer <- 
                                       coefkeras(self$model)
                                   }else{ 
                                     self$weights_last_layer <-
                                       cbind(self$weights_last_layer, 
                                             coefkeras(self$model))
                                   }
                                   self$on_metrics(logs, 0.1)
                                   
                                 },
                                 
                                 on_metrics = function(logs, sleep) {
                                   
                                   # record metrics
                                   for (metric in names(self$metrics)) {
                                     # guard against metrics not yet available by using NA 
                                     # when a named metrics isn't passed in 'logs'
                                     if(metric=="weight_diff"){
                                       nc <- NCOL(self$weights_last_layer)
                                       if(nc==1){
                                         value <- NA
                                       }else{
                                         value <- sum(abs(
                                           apply(self$weights_last_layer[,nc+(-1:0),drop=FALSE], 
                                                 1, diff)
                                         ))
                                       }
                                       if(is.null(value))
                                         value <- NA
                                     }else{
                                       value <- logs[[metric]]
                                       if (is.null(value))
                                         value <- NA
                                       else
                                         value <- mean(value)
                                     }
                                     self$metrics[[metric]] <- c(self$metrics[[metric]], value)
                                   }
                                   
                                   # create history object and convert to metrics data frame
                                   history <- keras:::keras_training_history(self$params, 
                                                                             self$metrics)
                                   metrics <- self$as_metrics_df(history)
                                   
                                   # view metrics if requested
                                   if (self$view_metrics) {
                                     
                                     # create the metrics_viewer or update if we already have one
                                     if (is.null(self$metrics_viewer)) {
                                       self$metrics_viewer <- tfruns::view_run_metrics(metrics)
                                     }
                                     else {
                                       tfruns::update_run_metrics(self$metrics_viewer, metrics)
                                     }
                                     
                                     # pump events
                                     Sys.sleep(sleep)
                                   }
                                   
                                   # record metrics
                                   tfruns::write_run_metadata("metrics", metrics)
                                   
                                 },
                                 
                                 # convert keras history to metrics data frame suitable for plotting
                                 as_metrics_df = function(history) {
                                   
                                   # create metrics data frame
                                   df <- as.data.frame(history$metrics)
                                   
                                   # pad to epochs if necessary
                                   pad <- history$params$epochs - nrow(df)
                                   pad_data <- list()
                                   for (metric in history$params$metrics)
                                     pad_data[[metric]] <- rep_len(NA, pad)
                                   df <- rbind(df, pad_data)
                                   
                                   # return df
                                   df
                                 },
                                 
                                 write_params = function(params) {
                                   properties <- list()
                                   properties$samples <- params$samples
                                   properties$validation_samples <- params$validation_samples
                                   properties$epochs <- params$epochs
                                   properties$batch_size <- params$batch_size
                                   tfruns::write_run_metadata("properties", properties)
                                 },
                                 
                                 write_model_info = function(model) {
                                   tryCatch({
                                     model_info <- list()
                                     model_info$model <- py_str(model, line_length = 80L)
                                     if (is.character(model$loss))
                                       model_info$loss_function <- model$loss
                                     else if (inherits(model$loss, "python.builtin.function"))
                                       model_info$loss_function <- model$loss$`__name__`
                                     optimizer <- model$optimizer
                                     if (!is.null(optimizer)) {
                                       model_info$optimizer <- py_str(optimizer)
                                       model_info$learning_rate <- k_eval(optimizer$lr)                     
                                     }
                                     tfruns::write_run_metadata("properties", model_info)
                                   }, error = function(e) {
                                     warning("Unable to log model info: ", e$message, call. = FALSE)
                                   })
                                   
                                 }
                               )
)

normalize_callbacks_with_metrics_custom <- function (view_metrics, callbacks)
{
  if (!is.null(callbacks) && !is.list(callbacks))
    callbacks <- list(callbacks)
  callbacks <- append(callbacks, KerasMetricsCallback_custom$new(view_metrics))
  keras:::normalize_callbacks(callbacks)
}
