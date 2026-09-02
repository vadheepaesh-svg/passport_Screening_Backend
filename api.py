import tempfile
import os
import shutil
import traceback
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from passport_full_pipeline import run_full_pipeline

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/screen-document")
async def screen_document(file: UploadFile = File(...)):
    temp_path = os.path.join(tempfile.gettempdir(), file.filename)

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = run_full_pipeline(temp_path)
        return result

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)  # clean up the uploaded file after processing


@app.get("/health")
async def health():
    return {"status": "ok"}