# Experiment Class

The Experiment object is created by passing the <i>params</i> dictionary containing all Dataset Locators. Using this object, you can run data pre-processing, training, validation, testing and prediction on your models. You can also reset the session without having to restart the cluster. Unlike the ETLSpec and MinibatchSpec classes, the arguments to all functions of the Experiment class are to be provided by the user. None of the functions in this class returns any value.
This class has the following functions - 

1. <b>run_etl</b>: This function is for running data pre-processing on your datasets. 
<br/>
<i>Arguments</i>: <br/>
   - imagenet_etl_spec - an instance/object of your implementation of the ETLSpec class.
   - fraction - an optional argument indicating fraction of the dataset that is to be used for the experiment. This is must be a float value between 0 and 1. It defaults to 1.

2. <b>run_fit</b>: This function call begins the model train and validation phase. The model's metrics are plotted on Tensorboard automatically.
<br/>
<i>Arguments</i>: <br/>
   - minibatch_spec - an instance/object of your implementation of the MinibatchSpec class.
   - param_grid - a dictionary of hyperparameters and the list of values to explore for each hyperparameter. This forms the model selection search space. A parameter named “batch_size” is mandatory.
   - num_epochs - the number of epochs the models have to be trained for. This is uniform for all models in the search space.

3. <b>run_test</b>: This function call is to run a specific model over the test dataset. The test metrics are printed as output.
<br/>
<i>Arguments</i>: <br/>
   - minibatch_spec - an instance/object of your implementation of the MinibatchSpec class.
   - model_tag - The ID of the model to be used for test. The ID can be found in the output of <i>run_fit</i>. This parameter can also be the model file's relative path (relative to <i>models dir</i>) of one of the models in <i>models dir</i> Dataset Locator. 
   - batch_size - the batch size to be used for running test

3. <b>run_predict</b>: This function call is to run a specific model over the inference dataset. The prediction values are presented as a downloadable .csv file.
<br/>
<i>Arguments</i>: <br/>
   - minibatch_spec - an instance/object of your implementation of the MinibatchSpec class.
   - model_tag - The ID of the model to be used for inference. The ID can be found in the output of <i>run_fit</i>. This parameter can also be the model file's relative path (relative to <i>models dir</i>) of one of the models in <i>models dir</i> Dataset Locator. 
   - batch_size - the batch size to be used for running test

