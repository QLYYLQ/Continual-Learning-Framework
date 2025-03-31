from munch import Munch
import yaml

def modify_default_cfg(default_config, main_cfg):
    for key, value in main_cfg.items():
        if isinstance(value, dict) and key in default_config and isinstance(default_config[key], dict):
            modify_default_cfg(default_config[key], value)
        else:
            default_config[key] = value


def load_yaml_file(filepath):
    with open(filepath, 'r',encoding="utf-8") as file:
        return yaml.safe_load(file)


def get_config(config_path,link_text=None):
    # Get the main configuration file
    main_cfg = load_yaml_file(config_path)

    # Get the default arguments, which are paths to other YAML files
    default_args = main_cfg.pop(link_text, None)

    if default_args is not None:
        # Load the configuration from the default argument files
        for path in default_args:
            default_config = load_yaml_file(path)

            # Modify content of default args if specified so in main configuration and then update main configuration with modified default configuration
            for key in main_cfg:
                if key in default_config:
                    modify_default_cfg(default_config[key], main_cfg[key])

            main_cfg.update(default_config)
    return Munch.fromDict(main_cfg)

if __name__ == '__main__':
    config_path = r"F:\Code_Field\Python_Code\Pycharm_Code\dataset\my_dataset\config\dataset\ade.yaml"
    config = get_config(config_path,"test")
    print(config)