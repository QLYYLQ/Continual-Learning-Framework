from utils.load_config.base_load import get_config
from pathlib import Path

model_config_root =Path(__file__).resolve().parent.parent.parent.joinpath('config', 'model')
module_part = ["image_encoder","text_encoder","position_embedding","query_net","mask_decoder"]


def get_model_config(model_name):
    init_model_config_path = model_config_root.joinpath(model_name+".yaml")
    # print(init_model_config_path)
    init_model_config = get_config(init_model_config_path)
    for module in module_part:
        if init_model_config[module].name == "none":
            setattr(init_model_config[module],"setting",{})
            continue
        module_path = model_config_root.joinpath(module,init_model_config[module].name+".yaml")
        init_module_config = get_config(module_path)

        setattr(init_model_config[module],"setting",init_module_config)
    return init_model_config

if __name__ == '__main__':
    config = get_model_config("model")
    print(config)