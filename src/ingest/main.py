from pathlib import Path
from typing import List
from llama_index.core import SimpleDirectoryReader, Document
from llama_index.core.readers.file.base import default_file_metadata_func
import base64
import tempfile
import os


def ingest_document(filename: str | None, content: str):
    # Decode content to temporary file, using original filename for filetype inference, if available
    file_content = base64.b64decode(content)

    tmpdir = tempfile.mkdtemp()

    if filename is None or filename == "":
        filename = "document"

    path = os.path.join(tmpdir, filename)

    with open(path, "wb") as file:
        file.write(file_content)

    # Cleanup
    os.rmdir(tmpdir)


def load_document(path: str) -> List[Document]:
    """Load a document from a given path.

    Args:
        path (str): Path to the document

    Returns:
        Document: Document object
    """
    return SimpleDirectoryReader.load_file(
        input_file=Path(path),
        file_metadata=default_file_metadata_func,
        filename_as_id=True,
        file_extractor={},
        errors="ignore",
        encoding="utf-8",
    )
