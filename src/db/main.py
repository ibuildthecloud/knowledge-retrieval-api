import psycopg2
from typing import List
from config import settings
from .errors import DatasetExistsError, DatasetDoesNotExistError
from llama_index.vector_stores.postgres import PGVectorStore


from llama_index.core import Document, get_response_synthesizer


from llama_index.core.indices import VectorStoreIndex

from llama_index.embeddings.openai import OpenAIEmbedding, OpenAIEmbeddingModelType

from llama_index.core.retrievers import VectorIndexRetriever

from llama_index.core.query_engine import RetrieverQueryEngine


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


def vector_store_exists(name: str) -> bool:
    with dbconn.cursor() as c:
        n = f"data_{name}"
        c.execute(
            "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = %s)",
            (n,),
        )
        x = c.fetchone()
        if x is None or not x[0]:
            print(f"Table {n} does not exist")
            return False
        return True


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
    if vector_store_exists(name):
        raise DatasetExistsError(name)

    # Initialize VectorStore for the dataset
    vector_store = get_vector_store(name, embed_dim=embed_dim)
    if isinstance(vector_store, PGVectorStore):
        vector_store._initialize()


def delete_dataset(name: str):
    """Deletes a VectorDB Dataset with the given name but errors if the dataset does not exist.

    Args:
        name (str): Name of the dataset

    Raises:
        Exception: Error if the dataset does not exist
    """

    # Raise Exception if dataset does not exist
    if not vector_store_exists(name):
        raise DatasetDoesNotExistError(name)

    with dbconn.cursor() as c:
        n = f"data_{name}"
        # Drop the table
        c.execute(f"DROP TABLE {n}")
        dbconn.commit()


def ingest_documents(
    dataset: str,
    documents: List[Document],
    embed_model_name: str = OpenAIEmbeddingModelType.TEXT_EMBED_ADA_002,
):
    if not vector_store_exists(dataset):
        raise DatasetDoesNotExistError(dataset)

    vector_store = get_vector_store(dataset)

    embed_model = OpenAIEmbedding(
        model=embed_model_name,
        # dimensions=vector_store.embed_dim, # FIXME: set dimensions only for models that support it
        # TODO: Set API parameters and allow for other embed_models to make it work with Rubra
    )

    vector_store_index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store, embed_model=embed_model
    )

    for document in documents:
        vector_store_index.insert(document)


def query(
    dataset: str,
    prompt: str,
    topk: int,
    embed_model_name: str = OpenAIEmbeddingModelType.TEXT_EMBED_ADA_002,
):
    if not vector_store_exists(dataset):
        raise DatasetDoesNotExistError(dataset)

    vector_store = get_vector_store(dataset)
    embed_model = OpenAIEmbedding(
        model=embed_model_name,
    )

    vector_store_index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store, embed_model=embed_model
    )

    retriever = VectorIndexRetriever(
        index=vector_store_index, embed_model=embed_model, similarity_top_k=topk
    )

    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=get_response_synthesizer(),  # TODO: optimize response_mode
    )

    return str(query_engine.query(prompt))  # TODO: use pydantic response
