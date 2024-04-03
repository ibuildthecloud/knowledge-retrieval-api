import yaml
import logging
import logging.config
from config import settings

log = logging.getLogger("knowledge-retrieval-api")


def init_logging():
    with open(settings.logging_conf_path, "r") as f:
        config = yaml.safe_load(f)
        logging.config.dictConfig(config)
