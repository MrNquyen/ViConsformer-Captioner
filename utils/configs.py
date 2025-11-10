from utils.registry import registry

class Config():
    def __init__(self, config):
        self.config_base = config
        self.build_config()

    def build_config(self):
        self.config_dataset = self.config_base["dataset_attributes"]
        self.config_model = self.config_base["model_attributes"]
        self.config_optimizer = self.config_base["optimizer_attributes"]
        self.config_training = self.config_base["training_parameters"]
    

    def build_registry(self):
        registry.set_module("config", name="model_attributes", instance=self.config_model)
        registry.set_module("config", name="dataset_attributes", instance=self.config_dataset)
        registry.set_module("config", name="optimizer_attributes", instance=self.config_optimizer)
        registry.set_module("config", name="training_parameters", instance=self.config_training)
        registry.set_module("config", name="common", instance=self.config_base)
            