import os
from typing import List
from pydantic import BaseModel

from .errors import DatasetDoesNotExistError
from .main import get_vector_store, vector_store_exists


from llama_index.core import get_response_synthesizer


from llama_index.core.indices import VectorStoreIndex

from llama_index.embeddings.openai import OpenAIEmbedding, OpenAIEmbeddingModelType


from llama_index.core.retrievers import VectorIndexRetriever

from llama_index.core.query_engine import RetrieverQueryEngine


class QueryResponseSourceNode(BaseModel):
    filename: str
    filetype: str
    page: str
    last_modified_date: str
    document_title: str


class QueryResponse(BaseModel):
    response: str
    sources: List[QueryResponseSourceNode] | None


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

    response = query_engine.query(prompt)

    sources: List[QueryResponseSourceNode] = [
        QueryResponseSourceNode(
            filename=os.path.basename(node.metadata.get("file_name", "")),
            filetype=node.metadata.get("file_type", ""),
            page=node.metadata.get("page_label", ""),
            last_modified_date=node.metadata.get("last_modified_date", ""),
            document_title=node.metadata.get("document_title", ""),
        )
        for node in response.source_nodes
    ]

    return QueryResponse(response=str(response), sources=sources)
