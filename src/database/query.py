import os
from typing import List
from pydantic import BaseModel

from config import settings

from .errors import DatasetDoesNotExistError
from .main import get_vector_store, dataset_exists


from llama_index.core.indices import VectorStoreIndex

from llama_index.embeddings.openai import OpenAIEmbedding, OpenAIEmbeddingModelType

from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import CitationQueryEngine


class QueryResponseSourceNode(BaseModel):
    filename: str
    filetype: str
    page: str
    last_modified_date: str
    document_title: str
    content: str
    all_metadata: dict


class QueryResponse(BaseModel):
    response: str
    sources: List[QueryResponseSourceNode] | None


def query(
    dataset: str,
    prompt: str,
    topk: int,
    embed_model_name: str = OpenAIEmbeddingModelType.TEXT_EMBED_ADA_002,
) -> QueryResponse | None:
    if not dataset_exists(dataset):
        raise DatasetDoesNotExistError(dataset)

    vector_store = get_vector_store(dataset)
    embed_model = OpenAIEmbedding(
        model=embed_model_name,
        api_base=settings.api_base,
        # additional_kwargs={"encoding_format": "float"},
    )

    vector_store_index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store, embed_model=embed_model
    )

    retriever = VectorIndexRetriever(
        index=vector_store_index, embed_model=embed_model, similarity_top_k=topk
    )

    citation_query_engine = CitationQueryEngine.from_args(
        index=vector_store_index,
        retriever=retriever,
        similarity_top_k=topk,
        embed_model=embed_model,
        citation_chunk_size=512,
    )

    citation_response = citation_query_engine.query(prompt)

    sources: List[QueryResponseSourceNode] = [
        QueryResponseSourceNode(
            filename=os.path.basename(node.metadata.get("file_name", "")),
            filetype=node.metadata.get("file_type", ""),
            page=node.metadata.get("page_label", ""),
            last_modified_date=node.metadata.get("last_modified_date", ""),
            document_title=node.metadata.get("document_title", ""),
            content=node.get_text(),
            all_metadata={
                k: v
                for k, v in node.metadata.items()
                if not k.startswith("_")
                and k
                not in [
                    "excluded_embed_metadata_keys",
                    "excluded_llm_metadata_keys",
                    "relationships",
                    "document_id",
                    "doc_id",
                    "ref_doc_id",
                ]
            },
        )
        for node in citation_response.source_nodes
    ]

    sources_txt = "\n".join(
        [
            f"[Source #{idx}] filename='{source.filename}' metadata={source.all_metadata}"
            for idx, source in enumerate(sources)
        ]
    )

    print(
        f"""
            Response:
            {citation_response}

            Sources:
            {sources_txt}
          """
    )

    return QueryResponse(response=str(citation_response), sources=sources)
