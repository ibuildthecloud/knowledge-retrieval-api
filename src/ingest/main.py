import time
from typing import List, Optional
from llama_index.core import SimpleDirectoryReader, Document
from llama_index.readers.file.pymu_pdf import PyMuPDFReader
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

    with open(path, "wb") as file:
        file.write(file_content)

    # Load file
    start = time.time()
    documents = load_file(path)
    log.debug(
        f"Loaded {len(documents)} documents from file {path} in {time.time() - start:.2f}s"
    )

    # Sanitize metadata, since e.g. modification date is wrong, as it was just created as a temp file
    # and keeping this metadata will screw with the cache key calculation
    for document in documents:
        # Dates are always the current date, so drop them
        for key in ["creation_date", "last_modified_date"]:
            document.metadata.pop(key, None)

        # Path is always tmpdir + filename, so remove the tmpdir part
        document.metadata["file_path"] = os.path.basename(
            document.metadata["file_path"]
        )

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


# Setting up a new file_extractor dict to replace some of the default readers with more performant ones
pdf_reader = PyMuPDFReader()  # PyMuPDFReader is way faster than the default PDFReader

file_extractor = SimpleDirectoryReader.supported_suffix_fn()
file_extractor[".pdf"] = pdf_reader


def load_file(path: str) -> List[Document]:
    """Load a document from a given path.

    Args:
        path (str): Path to the document

    Returns:
        List[Document]: Document objects
    """
    return SimpleDirectoryReader(
        input_files=[path], file_extractor=file_extractor
    ).load_data()
