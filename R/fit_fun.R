# "export" some keras functions
is_tensorflow_dataset <- keras:::is_tensorflow_dataset
resolve_view_metrics <- keras:::resolve_view_metrics
as_nullable_integer <- keras:::as_nullable_integer
as_class_weight <- keras:::as_class_weight
resolve_tensorflow_dataset <- keras:::resolve_tensorflow_dataset
to_keras_training_history <- keras:::to_keras_training_history
write_history_metadata <- keras:::write_history_metadata
keras_version <- keras:::keras_version

# copy of keras:::fit.keras.engine.training.Model
fit_fun <- function(object, x = NULL, y = NULL, batch_size = NULL, epochs = 10, 
                    verbose = getOption("keras.fit_verbose", default = 1), callbacks = NULL, 
                    view_metrics = getOption("keras.view_metrics", default = "auto"), 
                    validation_split = 0, validation_data = NULL, shuffle = TRUE, 
                    class_weight = NULL, sample_weight = NULL, initial_epoch = 0, 
                    steps_per_epoch = NULL, validation_steps = NULL, ...)
{
  
  if (is.null(batch_size) && is.null(steps_per_epoch) && !is_tensorflow_dataset(x)) 
    batch_size <- 32L
  if (identical(view_metrics, "auto")) 
    view_metrics <- resolve_view_metrics(verbose, epochs, 
                                         object$metrics)
  args <- list(batch_size = as_nullable_integer(batch_size), 
               epochs = as.integer(epochs), verbose = as.integer(verbose), 
               callbacks = normalize_callbacks_with_metrics_custom(view_metrics, 
                                                            callbacks), 
               validation_split = validation_split, 
               shuffle = shuffle, class_weight = as_class_weight(class_weight), 
               sample_weight = keras_array(sample_weight), initial_epoch = as.integer(initial_epoch))
  if (!is.null(validation_data)) {
    dataset <- resolve_tensorflow_dataset(validation_data)
    if (!is.null(dataset)) 
      args$validation_data <- dataset
    else args$validation_data <- keras_array(validation_data)
  }
  dataset <- resolve_tensorflow_dataset(x)
  if (inherits(dataset, "tensorflow.python.data.ops.dataset_ops.DatasetV2")) {
    args$x <- dataset
  }
  else if (!is.null(dataset)) {
    args$x <- dataset[[1]]
    args$y <- dataset[[2]]
  }
  else {
    if (!is.null(x)) 
      args$x <- keras_array(x)
    if (!is.null(y)) 
      args$y <- keras_array(y)
  }
  if (keras_version() >= "2.0.7") {
    args$steps_per_epoch <- as_nullable_integer(steps_per_epoch)
    args$validation_steps <- as_nullable_integer(validation_steps)
  }
  history <- do.call(object$fit, args)
  history <- to_keras_training_history(history)
  write_history_metadata(history)
  invisible(history)
  
}