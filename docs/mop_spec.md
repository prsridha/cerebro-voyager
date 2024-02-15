# Minibatch Specification

Similar to the ETL Spec, the Minibatch Spec class has functions which, when filled with code, instruct Cerebro on how to build, train and evaluate your models. Your can refer to the implementation given [here](../examples/Resnet%20on%20Imagenet/cerebro_imagenet.ipynb) for Resnet50 on Imagenet as an example. 
Also, we recommend that all the Python imports that a particular function needs be specified within the function body, and not outside it. The Minibatch Spec class can contain any other functions defined by the user, called from within the functions given below. 
The Minibatch Spec class has the following functions.

1. <b>initialize worker()</b>: This function is used for setup of packages or for other one-time-per-worker tasks. This function will be executed exactly once on each worker. It can be left blank if no such tasks exist. For example - if you task involves NLP, you would add code to download tokenizer models from NLTK here.   
2. <b>read_misc</b>: This function is for accessing the files specified under the <i>misc</i> field in <i>params</i>. 
<br/>
<i>Arguments</i>: <br/>
   - misc_path - The directory path in which all <i>misc</i> files present. The files can be accessed by appending the file names to this path.
    <br/>

   <i>Returns</i>: None
<br/>
3. <b>create_model_components:</b> This function is to create models for model selection using the hyperparameter combination that Cerebro generates.   

    <i>Arguments</i>: <br/>
   - hyperparams - a single hyperparameter configuration dictionary, picked from all possible combinations of <i>param grid</i>. 
    <br/>

   <i>Returns</i>: A dictionary mapping the model object names to the model objects. Here, the model objects refer to the entire model saved as a file (not just the state_dict). The dictionary should also contain a key named “optimizer” pointing to the optimizer object.

4. <b>metrics_agg:</b> This function is to aggregate metrics that have been generated at each iteration of the train, val and test operations.
<br/>
<i>Arguments</i>: <br/>
   - mode - this string can be one of "train", "val", "test" or "predict", depending on the current operation.
   - hyperparams: a single hyperparameter configuration dictionary, for the model in question.
   - metrics: this is a dictionary whose keys are the same as the ones returned by the <i>train</i> function or the <i>valtest</i> function. The dictionary's values are a list of accumulated metrics from each iteration of the <i>train</i> function or the <i>valtest</i> function. 
    <br/>

   <i>Returns</i>: For train, val and test modes, the function must return a tuple of two PyTorch Tensors - the processed Tensor and the label Tensor. For predict mode, the function must return a tuple of the processed tensor and None.
5. <b>train:</b> This function is to train the model(s) on a single minibatch of data.
<br/>
<i>Arguments</i>: <br/>
   - model_object - the model object that was created in create_model_components
   - minibatch: a single minibatch of data from the train dataset, based on the batch_size hyperparameter value
   - hyperparams: a single hyperparameter configuration dictionary, for the model in question
   - device: the Voayger's HPU device on which the model and data should be moved to 
    <br/>

   <i>Returns</i>: A dictionary containing minibatch level metrics. The keys of the dictionary must describe the metric (such as 'loss' or 'top-5 accuracy') and the values in the dictionary must be single values (such as Float or Tensor, not List or Dict)
6. <b>valtest:</b> This function is to run the model(s) on the validation and test dataset. The same function is used for both validation and test operations.
<br/>
<i>Arguments</i>: <br/>
   - model_object - the model object that was created in create_model_components
   - minibatch: a single minibatch of data from the train dataset, based on the batch_size hyperparameter value
   - hyperparams: a single hyperparameter configuration dictionary, for the model in question
   - device: the Voayger's HPU device on which the model and data should be moved to 
    <br/>

   <i>Returns</i>: A dictionary containing validation or test metrics. The keys of the dictionary must describe the metric (such as 'loss' or 'top-5 accuracy') and the values in the dictionary must be single values (such as Float or Tensor, not List or Dict)
7. <b>predict:</b> This function is to run the model(s) on the inference dataset.
<br/>
<i>Arguments</i>: <br/>
   - model_object - the model object that was created in create_model_components
   - minibatch: a single minibatch of data from the train dataset, based on the batch_size hyperparameter value
   - hyperparams: a single hyperparameter configuration dictionary, for the model in question
   - device: the Voayger's HPU device on which the model and data should be moved to 
    <br/>

   <i>Returns</i>: Two lists - one that contains the predicted classes and the other that contains confidence probabilities for each of the examples.
<br/>

<br/><br/>
The Minibatch Spec class template -

![mop_spec](img/mop_spec.png)