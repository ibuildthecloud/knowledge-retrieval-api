import psycopg2

from config import settings
from .errors import DatasetExistsError, DatasetDoesNotExistError
from llama_index.vector_stores.postgres import PGVectorStore

#
# Database Connection
#
dbconn = psycopg2.connect(
    host=settings.db_host,
    port=settings.db_port,
    dbname=settings.db_dbname,
    user=settings.db_user,
    password=settings.db_password,
)
dbconn.autocommit = True


def get_vector_store(name: str, embed_dim: int = 1536) -> PGVectorStore:
    """Get a VectorStore for the given dataset name and embedding dimension

    Args:
        name (str): Name of the dataset
        embed_dim (int, optional): Embedding Dimension. Defaults to 1536, which is the OpenAI embedding dimension.

    Returns:
        PGVectorStore: VectorStore for the given dataset name and embedding dimension
    """
    return PGVectorStore.from_params(
        host=settings.db_host,
        port=str(settings.db_port),
        database=settings.db_dbname,
        user=settings.db_user,
        password=settings.db_password,
        embed_dim=embed_dim,
        table_name=name,
    )


def create_dataset(name: str, embed_dim: int = 1536):
    """Creates and initializes a new VectorDB Dataset with the given name and embedding dimension but errors if the dataset already exists.

    Args:
        name (str): Name of the dataset
        embed_dim (int, optional): Embedding Dimension. Defaults to 1536, which is the OpenAI embedding dimension.

    Raises:
        Exception: Error if the dataset already exists
    """

    # Raise Exception if dataset already exists
    with dbconn.cursor() as c:
        n = f"data_{name}"
        print(n)
        c.execute(
            "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = %s)",
            (n,),
        )
        x = c.fetchone()
        if x is not None and x[0]:
            raise DatasetExistsError(name)

    # Initialize VectorStore for the dataset
    get_vector_store(name, embed_dim=embed_dim)._initialize()


def delete_dataset(name: str):
    """Deletes a VectorDB Dataset with the given name but errors if the dataset does not exist.

    Args:
        name (str): Name of the dataset

    Raises:
        Exception: Error if the dataset does not exist
    """

    # Raise Exception if dataset does not exist
    with dbconn.cursor() as c:
        n = f"data_{name}"
        c.execute(
            "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = %s)",
            (n,),
        )
        x = c.fetchone()
        if x is None or not x[0]:
            raise DatasetDoesNotExistError(name)

        # Drop the table
        c.execute(f"DROP TABLE {n}")
        dbconn.commit()
