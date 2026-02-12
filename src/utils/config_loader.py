import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    """
    Loads YAML configuration file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Parsed config as a dictionary
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
