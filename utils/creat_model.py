from utils.load_model.load_model_config import get_model_config,module_part
from model.base_model import BaseModel
from model.utils.register import model_entrypoint


def load_model_from_config(model_config):
    model = BaseModel()
    for module in module_part:
        build_function = model_entrypoint(module[module].name)
        setattr(model,module,build_function(model_config[module].setting))
