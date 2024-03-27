import os
from pydantic_settings import BaseSettings

config_dir = os.path.dirname(os.path.abspath(__file__))


class Settings(BaseSettings):
    db_file_path: str = os.path.join(
        os.path.dirname(__file__), "kra.db"
    )  # FIXME: XDG_DATA_HOME

    api_base: str = "http://localhost:8080/v1/"

    alembic_ini_path: str = os.path.join(config_dir, "alembic.ini")
    logging_conf_path: str = os.path.join(config_dir, "log_conf.yaml")

    debug: bool = False

    class Config:
        env_prefix = "KRA_"


settings = Settings()
