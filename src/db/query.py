from .errors import DatasetDoesNotExistError
from .main import get_vector_store, vector_store_exists


from llama_index.core import get_response_synthesizer


from llama_index.core.indices import VectorStoreIndex

from llama_index.embeddings.openai import OpenAIEmbedding, OpenAIEmbeddingModelType


from llama_index.core.retrievers import VectorIndexRetriever

from llama_index.core.query_engine import RetrieverQueryEngine


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
