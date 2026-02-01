import logging
import logging.config

import yaml
import os

yaml_path = os.path.join(os.path.dirname(__file__), "logging_config.yaml")


def setup_logging(yaml_path: str = yaml_path):
    """Setup logging from YAML configuration file."""

    try:
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)

        logging.config.dictConfig(config)
    except FileNotFoundError:
        print(f"Error: YAML file not found at {yaml_path}")
    except yaml.YAMLError as e:
        print(f"Error: YAML file is invalid: {e}")


if __name__ == "__main__":
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Logging setup complete")
