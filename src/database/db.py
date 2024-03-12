from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import declarative_base, sessionmaker, scoped_session
from sqlalchemy import URL
from config import settings

engine: Engine | None = None

# Use scoped_session to ensure thread safety
SessionLocal: scoped_session | None = None


Base = declarative_base()

db_url = URL.create(
    drivername="postgresql+psycopg2",
    host=settings.db_host,
    port=settings.db_port,
    username=settings.db_user,
    password=settings.db_password,
    database=settings.db_dbname,
)


def get_engine():
    global engine
    if engine is None:
        engine = create_engine(db_url, pool_pre_ping=True)
    return engine


def get_session() -> scoped_session:
    global SessionLocal
    if SessionLocal is None:
        SessionLocal = scoped_session(
            sessionmaker(autocommit=False, autoflush=False, bind=get_engine())
        )
    return SessionLocal()


#
# MODELS
#

from database.models import DocumentIndex, FileIndex  # noqa


def init_db():
    """Initialize database"""
    Base.metadata.create_all(bind=get_engine())
