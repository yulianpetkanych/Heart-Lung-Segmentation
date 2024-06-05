from typing import List
from fastapi import BackgroundTasks, FastAPI, Request, UploadFile, File
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import zipfile
from io import BytesIO
from fastapi.staticfiles import StaticFiles
import uvicorn

from heart_final import heart_segmentation
from lungmask_final import lung_segmentation

app = FastAPI()

results_directory = "results"
upload_directory = "uploaded_files"
heart_npy_path = "heart_npy"
heart_jpg_path = "heart_jpg"
heart_model_path = "last_epoch_model.pth"

# Дозволяємо CORS для всіх джерел (або налаштуйте відповідно до ваших потреб)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Монтуємо папку зі статичними файлами
app.mount("/static", StaticFiles(directory="static"), name="static")


def clean_up_files(directories: List[str]):
    for directory in directories:
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error: {e}")

# Головна сторінка
@app.get("/")
def read_root():
    clean_up_files([upload_directory])
    return FileResponse("index.html")

@app.middleware("http")
async def clean_up_files_middleware(request: Request, call_next):
    if request.url.path == "/static/index.html":
        clean_up_files([upload_directory])
    response = await call_next(request)
    return response

# Маршрут для завантаження файлів
@app.post("/uploads")
async def upload_files(files: List[UploadFile] = File(...)):
    os.makedirs(upload_directory, exist_ok=True)
    
    for file in files:
        file_location = os.path.join(upload_directory, file.filename)
        with open(file_location, "wb+") as file_object:
            file_object.write(file.file.read())

    return JSONResponse(content={"message": "Files uploaded successfully"})


# Маршрут для отримання зображень на основі button_id
@app.get("/get_images")
async def get_images(button_id: str, background_tasks: BackgroundTasks):
    # Ваш логічний код для отримання зображень на основі button_id
    # В цьому випадку ми використовуємо папку results
    print(button_id)
    if button_id == "heart":
        print("gay")
        _ = heart_segmentation(root_path=upload_directory,
                               path_to_save=results_directory,
                               organ_num=1,
                               path_for_npy=heart_npy_path,
                               path_for_jpg=heart_jpg_path,
                               model_path=heart_model_path)
        # for file in os.listdir(upload_directory):
        #     file_location = os.path.join(upload_directory, file)
        #     heart_segmentation(file_location, results_directory)
    elif button_id == "lung_1":
        _ = heart_segmentation(root_path=upload_directory,
                               path_to_save=results_directory,
                               organ_num=0,
                               path_for_npy=heart_npy_path,
                               path_for_jpg=heart_jpg_path,
                               model_path=heart_model_path)

    elif button_id == "lung_2":
        _ = lung_segmentation(root_path=upload_directory,
                               path_to_save=results_directory
                               )


    background_tasks.add_task(clean_up_files, [heart_npy_path, heart_jpg_path, results_directory])
    memory_file = BytesIO()

    with zipfile.ZipFile(memory_file, 'w') as zf:
        for root, _, files in os.walk(results_directory):
            for file in files:
                file_path = os.path.join(root, file)
                zf.write(file_path, arcname=os.path.relpath(file_path, results_directory))

    memory_file.seek(0)
    return StreamingResponse(memory_file, media_type="application/x-zip-compressed", headers={"Content-Disposition": "attachment; filename=images.zip"})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
