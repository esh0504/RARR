import importlib

def find_datasets_def(dataset_name):
    module_name = 'data.{}'.format(dataset_name)
    module = importlib.import_module(module_name)
    return getattr(module, "RainDataset")