import time
from typing import List, Optional
from llama_index.core import SimpleDirectoryReader, Document
import base64
import tempfile
import os

from log import log
from database import ingest_documents, get_session
from database.models import FileIndex, DocumentIndex

from database.errors import DocumentExistsError


async def ingest_file(
    dataset: str,
    content: str,
    file_id: str,
    filename: Optional[str] = None,
) -> dict:
    """Ingest a file into the VectorDB.

    Args:
        dataset (str): Name of the target dataset
        content (str): Base64 encoded file content
        filename (str | None): Name of the file, if available - makes filetype guessing more accurate
        file_id (str | None): Optional ID that will be used to identify documents generated from the file

    Returns:
        int: Number of ingested documents (a single file can be multiple documents)
    """
    db = get_session()

    if file_id is not None:
        existing = (
            db.query(FileIndex)
            .filter_by(dataset=dataset, file_id=file_id)
            .one_or_none()
        )
        if existing is not None:
            db.close()
            raise DocumentExistsError(dataset, file_id)

    # Decode content to temporary file, using original filename for filetype inference, if available
    file_content = base64.b64decode(content)

    tmpdir = tempfile.mkdtemp()

    if filename is None or filename == "":
        filename = "document"

    # TODO: use google's magika to guess filetype and filter supported

    path = os.path.join(tmpdir, filename)

    log.info(f"Writing file to {path}")
    with open(path, "wb") as file:
        file.write(file_content)

    # Load file
    log.info(f"Loading file from {path}")
    start = time.time()
    documents = load_file(path)
    log.info(f"Loaded {len(documents)} documents in {time.time() - start:.2f}s")

    # Sanitize metadata, since e.g. modification date is wrong, as it was just created as a temp file
    # and keeping this metadata will screw with the cache key calculation
    log.info("Sanitizing metadata")
    start = time.time()
    for document in documents:
        # Dates are always the current date, so drop them
        for key in ["creation_date", "last_modified_date"]:
            document.metadata.pop(key, None)

        # Path is always tmpdir + filename, so remove the tmpdir part
        document.metadata["file_path"] = os.path.basename(
            document.metadata["file_path"]
        )
    log.info(f"Sanitized metadata in {time.time() - start:.2f}s")

    # Ingest documents: Create embeddings and store in the VectorDB
    await ingest_documents(dataset, documents)

    # Cleanup
    os.remove(path)
    os.rmdir(tmpdir)

    doc_ids = [doc.doc_id for doc in documents]

    file_index = FileIndex(
        dataset=dataset,
        file_id=file_id,
    )

    docs: list[DocumentIndex] = [
        DocumentIndex(
            dataset=dataset,
            file_id=file_id,
            document_id=doc_id,
        )
        for doc_id in doc_ids
    ]

    db.add(file_index)
    db.commit()
    db.add_all(docs)
    db.commit()

    db.close()

    return {
        "num_ingested_docs": len(documents),
        "documents": doc_ids,
    }


def load_file(path: str) -> List[Document]:
    """Load a document from a given path.

    Args:
        path (str): Path to the document

    Returns:
        Document: Document object
    """
    return SimpleDirectoryReader(input_files=[path]).load_data()
