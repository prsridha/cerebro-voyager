class Parallelism:
    def __init__(self, worker_id, model_config, model_checkpoint_path, epoch):
        pass

    def save_local_metrics(self, rank, metrics, user_metrics_func):
        pass

    def load_checkpoint(self, model_object):
        pass

    def save_checkpoint(self, model_object):
        pass

    def execute_sample(self, user_train_func, dataset):
        pass

    def execute_train(self, dataset, model_id):
        pass

    def execute_val(self, dataset, model_id):
        pass

    def execute_test(self, dataset):
        pass

    def execute_predict(self, dataset):
        pass
