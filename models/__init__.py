import importlib

def find_models_def(model_name):
    module_name = 'models.{}'.format(model_name)
    module = importlib.import_module(module_name)
    return getattr(module, "Model")