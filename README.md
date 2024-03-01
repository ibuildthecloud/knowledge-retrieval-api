<p align="center">
  <img src="src/static/img/icon.png" />
</p>

# knowledge-retrieval-api

Standalone Knowledge Retrieval API Server to be used with Rubra

## Development

- Run in development mode (hot-reloading): `make run-dev` (Requires `docker` and `compose`)
- Dependency Management: `uv`
- Linting & Formatting: `ruff`

## File Types

Currently, the following file types are supported for ingestion via llama-index' [`SimpleDirectoryReader`](https://docs.llamaindex.ai/en/stable/module_guides/loading/simpledirectoryreader.html#supported-file-types) interface:

- `.csv` - comma-separated values
- `.docx` - Microsoft Word
- `.epub` - EPUB ebook format
- `.hwp` - Hangul Word Processor
- `.ipynb` - Jupyter Notebook
- `.jpeg`, .jpg - JPEG image
- `.mbox` - MBOX email archive
- `.md` - Markdown
- `.mp3, .mp4` - audio and video
- `.pdf` - Portable Document Format
- `.png` - Portable Network Graphics
- `.ppt, .pptm, .pptx` - Microsoft PowerPoint
