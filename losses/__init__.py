import importlib

def find_loss_def(loss_name):
    module_name = 'losses.{}'.format(loss_name)
    module = importlib.import_module(module_name)
    return getattr(module, "Loss")