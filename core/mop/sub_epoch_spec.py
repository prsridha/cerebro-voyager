class SubEpochSpec:
    def __init__(self):
        pass

    def initialize_worker(self):
        pass

    def create_model_components(self, hyperparams):
        pass

    def train(self, models, optimizer, checkpoint, dataloader, hyperparams, input_device, output_device):
        pass

    def valtest(self, models, checkpoint, dataloader, hyperparams, input_device, output_device):
        pass

    def predict(self, models, checkpoint, dataloader, hyperparams, input_device):
        pass
