class Parallelism:
    def __init__(self, worker_id, model_config, model_checkpoint_path):
        pass

    def save_metrics(self, rank, metrics, user_metrics_func):
        pass

    def load_checkpoint(self, model_object):
        pass

    def save_checkpoint(self, model_object):
        pass

    def execute_sample(self, user_train_func, dataset):
        pass

    def execute_train(self, user_train_func, user_metrics_func, dataset, metrics_cycle_size, model_id):
        pass

    def execute_val(self, user_val_func, user_metrics_func, dataset, model_id, epoch):
        pass

    def execute_test(self, user_test_func, dataset, output_path):
        pass

    def execute_predict(self, user_pred_func, dataset, output_path):
        pass
