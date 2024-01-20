class MiniBatchSpec:
    def __init__(self):
        pass

    def initialize_worker(self):
        pass

    def create_model_components(self, hyperparams):
        pass

    def train(self, model_object, minibatch, hyperparams, device):
        pass

    def valtest(self, model_object, minibatch, hyperparams, device):
        pass

    def predict(self, model_object, minibatch, hyperparams, device):
        pass
