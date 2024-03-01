from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.staticfiles import StaticFiles
import db
import ingest.main as ingest


app = FastAPI(title="Rubra - Knowledge Retrieval API")
app.mount("/static", StaticFiles(directory="src/static"), name="static")


#
# Pydantic Data Models
#
class Dataset(BaseModel):
    name: str
    embed_dim: int


class Query(BaseModel):
    prompt: str
    topk: int


class Ingest(BaseModel):
    filename: str | None  # Optional filename
    data: str  # Base64 encoded data


#
# Endpoints
#
@app.get("/", include_in_schema=False)
async def root() -> HTMLResponse:
    return HTMLResponse(
        content="""
        <html>
            <body>
                <p align="center">
                    <img src="/static/img/icon.png" />
                </p>
                <h1 align="center">Rubra - Knowledge Retrieval API</h1>
                <p align="center">Visit <a href="/docs">/docs</a> for the API documentation</p>
            </body>
        </html>
        """
    )


@app.get("/docs", include_in_schema=False)
async def swagger_ui_html():
    return get_swagger_ui_html(
        title=app.title + "- Swagger UI",
        openapi_url="/openapi.json",
        swagger_favicon_url="/static/img/favicon.ico",
        swagger_ui_parameters=app.swagger_ui_parameters,
    )


@app.get("/favicon.io", include_in_schema=False)
async def favicon() -> FileResponse:
    return FileResponse(path="/static/img/favicon.ico")


@app.post("/datasets/create")
async def create_dataset(dataset: Dataset) -> Dataset:
    """Endpoint to create a new dataset in the VectorDB.

    Args:
        name (str): Name of the dataset to create

    Raises:
        HTTPException: 409 Conflict if the dataset already exists
        HTTPException: 500 Internal Server Error if any other error occurs

    Returns:
        Dataset: Resulting dataset object
    """
    try:
        db.create_dataset(dataset.name)  # TODO: take embed_dim as input
        return Dataset(name=dataset.name, embed_dim=1536)
    except db.DatasetExistsError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/datasets/{name}")
async def delete_dataset(name: str):
    """Deletes a dataset from the VectorDB.

    Args:
        name (str): Name of the dataset to delete

    Raises:
        HTTPException: 404 Not Found if the dataset does not exist
        HTTPException: 500 Internal Server Error if any other error occurs

    Returns:
        JSONResponse: Success message on successful deletion
    """
    # Delete dataset from the VectorDB
    try:
        db.delete_dataset(name)
        return {"message": f"Dataset '{name}' deleted successfully"}
    except db.DatasetDoesNotExistError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/datasets/{name}/query")
async def query(name: str, query: str, topk: int = 5):
    """Query the VectorDB with a user-given prompt.

    Args:
        name (str): Target Dataset Name
        query (str): Prompt to query the dataset with

    Raises:
        HTTPException: 404 Not Found if the dataset does not exist
        HTTPException: 500 Internal Server Error if any other error occurs

    Returns:
        JSONResponse: Top-k results from the query
    """
    try:
        results = db.query(prompt=query, dataset=name, topk=topk)
        return {"results": results}
    except db.DatasetDoesNotExistError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/datasets/{name}/ingest")
async def ingest_data(name: str, input_data: Ingest):
    """Ingest new data into the VectorDB.

    Args:
        input (Ingest): Base64 encoded data to ingest, with an optional filename.

    Raises:
        HTTPException: 404 Not Found if the dataset does not exist
        HTTPException: 500 Internal Server Error if any other error occurs

    Returns:
        JSONResponse: Success message on successful ingestion
    """
    # Ingest new data into the VectorDB
    try:
        # Convert base64 data to appropriate format and ingest into pgvector
        # You would need to implement the logic here based on your requirements
        # This could involve using OpenAI API for embedding and then storing in pgvector
        ingested = await ingest.ingest_file(name, input_data.filename, input_data.data)
        return {
            "message": (
                "Successfully ingested data" + f" from file '{input_data.filename}'"
                if input_data.filename
                else ""
            ),
            "num_ingested_docs": ingested,
        }
    except db.DatasetDoesNotExistError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
