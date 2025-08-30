import yaml
from train import train_model

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    config = load_config("config/train_config.yaml")
    train_model(config)