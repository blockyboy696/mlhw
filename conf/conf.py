import logging
from dynaconf import Dynaconf

#specifying logging level
logging.basicConfig(level=logging.INFO)

settings = Dynaconf(settings_file = "settings.toml")