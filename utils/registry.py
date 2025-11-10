class Registry:
    mapping = {
        "args": None,
        "config": {
            "model_attributes": None,
            "dataset_attributes": None,
            "optimizer_attributes": None,
            "training_parameters": None,
            "common": None
        },
        "writer": {
            "common": None,
            "evaluation": None,
            "inference": None,
        }
    }

    @classmethod
    def register_writer(cls, name):
        r"""Register a writer to registry with key 'name'

        Args:
            name: Key with which the variable will be registered.

        Usage::

            from pythia.common.registry import registry
            from pythia.trainers.custom_trainer import CustomTrainer


            @registry.register_trainer("custom_trainer")
            class CustomTrainer():
                ...

        """
        def wrap(trainer_cls):
            cls.mapping["writer"][name] = trainer_cls
            return trainer_cls
        return wrap

    def set_module(self, parent, name=None, instance=None):
        if name==None:
            self.mapping[parent] = instance
        else:
            self.mapping[parent][name] = instance

    def get_writer(cls, name):
        return cls.mapping["writer"].get(name, None)
    
    def get_args(cls, name):
        return getattr(cls.mapping["args"], name)
    
    def get_config(cls, name):
        return cls.mapping["config"].get(name, None)

registry = Registry()