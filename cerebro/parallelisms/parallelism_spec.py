class Parallelism:
    def __init__(self, worker_id, model_config, model_checkpoint_path, epoch, sample_size=None):
        pass

    def save_local_metrics(self, rank, metrics, user_metrics_func):
        pass

    def load_checkpoint(self, model_object):
        pass

    def save_checkpoint(self, model_object):
        pass

    def execute_sample(self, minibatch_spec):
        pass

    def execute_train(self, minibatch_spec, model_id):
        pass

    def execute_val(self, minibatch_spec, model_id):
        pass

    def execute_test(self, minibatch_spec, model_tag):
        pass

    def execute_predict(self, minibatch_spec, model_tag):
        pass
