from typing import List
from llama_index.core import SimpleDirectoryReader, Document
import base64
import tempfile
import os

from db import ingest_documents


async def ingest_file(dataset: str, filename: str | None, content: str) -> int:
    """Ingest a file into the VectorDB.

    Args:
        dataset (str): Name of the target dataset
        filename (str | None): Name of the file, if available - makes filetype guessing more accurate
        content (str): Base64 encoded file content

    Returns:
        int: Number of ingested documents (a single file can be multiple documents)
    """

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
    documents = load_file(path)

    # Ingest documents: Create embeddings and store in the VectorDB
    await ingest_documents(dataset, documents)

    # Cleanup
    os.remove(path)
    os.rmdir(tmpdir)

    return len(documents)


def load_file(path: str) -> List[Document]:
    """Load a document from a given path.

    Args:
        path (str): Path to the document

    Returns:
        Document: Document object
    """
    return SimpleDirectoryReader(input_files=[path]).load_data()
