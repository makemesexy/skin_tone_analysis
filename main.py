from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np
import mediapipe as mp
import cv2
import aiohttp
import asyncio
from concurrent.futures import ThreadPoolExecutor
from sklearn.cluster import KMeans
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import stone

app = FastAPI()

# Configuration for hair color segmentation
base_options = python.BaseOptions(model_asset_path='hair_segmenter.tflite')
options = vision.ImageSegmenterOptions(base_options=base_options, output_category_mask=True)
BG_COLOR = (192, 192, 192)  # gray
MASK_COLOR = (255, 255, 255)  # white
DESIRED_HEIGHT = 512
DESIRED_WIDTH = 512
mp_face_mesh = mp.solutions.face_mesh
executor = ThreadPoolExecutor()

# Models for requests
class ImageRequest(BaseModel):
    url: str

# Utility functions
async def download_image(url: str) -> np.ndarray:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    image_data = await response.read()
                    image_data = np.frombuffer(image_data, np.uint8)
                    return cv2.imdecode(image_data, cv2.IMREAD_COLOR)
                else:
                    raise HTTPException(status_code=400, detail="Invalid image URL")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

def dominant_color(image, mask):
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    masked_image_rgb = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)
    pixels = masked_image_rgb.reshape((-1, 3))
    pixels = pixels[np.all(pixels != [0, 0, 0], axis=1)]
    kmeans = KMeans(n_clusters=1)
    kmeans.fit(pixels)
    return kmeans.cluster_centers_[0].astype(int)

def average_color(color1, color2):
    return tuple((color1 + color2) // 2)

def rgb_to_hex(color):
    return '#{:02x}{:02x}{:02x}'.format(color[0], color[1], color[2])

def resize_image(image):
    h, w = image.shape[:2]
    if h < w:
        img = cv2.resize(image, (DESIRED_WIDTH, int(h / (w / DESIRED_WIDTH))))
    else:
        img = cv2.resize(image, (int(w / (h / DESIRED_HEIGHT)), DESIRED_HEIGHT))
    return img

def find_dominant_color(image, mask):
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    hair_region = lab_image[mask].reshape(-1, 3)
    kmeans = KMeans(n_clusters=1)
    kmeans.fit(hair_region)
    dominant_color_lab = kmeans.cluster_centers_[0].astype('uint8').reshape(1, 1, 3)
    dominant_color_rgb = cv2.cvtColor(dominant_color_lab, cv2.COLOR_LAB2RGB).reshape(3,)
    return dominant_color_rgb

# API Endpoints
@app.post("/get_iris_colour")
async def get_iris_colour(request: ImageRequest):
    frame = await download_image(request.url)
    frame = cv2.flip(frame, 1)
    LEFT_IRIS = [474, 475, 476, 477]
    RIGHT_IRIS = [469, 470, 471, 472]

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_h, img_w = frame.shape[:2]
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) 
                                    for p in results.multi_face_landmarks[0].landmark])

            (l_cx, l_cy), l_radius = cv2.minEnclosingCircle(mesh_points[LEFT_IRIS])
            (r_cx, r_cy), r_radius = cv2.minEnclosingCircle(mesh_points[RIGHT_IRIS])
            center_left = np.array([l_cx, l_cy], dtype=np.int32)
            center_right = np.array([r_cx, r_cy], dtype=np.int32)

            mask_left = np.zeros((img_h, img_w), dtype=np.uint8)
            mask_right = np.zeros((img_h, img_w), dtype=np.uint8)

            cv2.circle(mask_left, tuple(center_left), int(l_radius), 255, -1)
            cv2.circle(mask_right, tuple(center_right), int(r_radius), 255, -1)

            dominant_color_left = dominant_color(frame, mask_left)
            dominant_color_right = dominant_color(frame, mask_right)

            final_color = average_color(dominant_color_left, dominant_color_right)
            final_hex_code = rgb_to_hex(final_color)

            return {"eye_colour": final_hex_code}
        else:
            return {"error": "No face landmarks detected."}


@app.post("/get_hair_colour")
async def get_hair_colour(request: ImageRequest):
    image = await download_image(request.url)
    if image is None:
        return JSONResponse(content={"error": "Invalid image file."}, status_code=400)

    image = resize_image(image)

    with vision.ImageSegmenter.create_from_options(options) as segmenter:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        segmentation_result = segmenter.segment(mp_image)
        category_mask = segmentation_result.category_mask

        fg_image = np.zeros(image.shape, dtype=np.uint8)
        fg_image[:] = MASK_COLOR
        bg_image = np.zeros(image.shape, dtype=np.uint8)
        bg_image[:] = BG_COLOR
        bg_image = bg_image[:, :, :3]

        condition = np.stack((category_mask.numpy_view(),) * 3, axis=-1) > 0.5
        output_image = np.where(condition, image, bg_image)

        dominant_color = find_dominant_color(image, condition)
        dominant_color_hex = rgb_to_hex(dominant_color)

        return {
            "hair_colour": dominant_color_hex
        }

@app.post("/get_skin_tone")
async def get_skin_tone(request: ImageRequest):
    result = stone.process(request.url, image_type="color", return_report_image=True)
    skin_tone = result["faces"][0]["skin_tone"]
    return {"skin_tone": skin_tone}


@app.post("/get_skin_analysis")
async def get_combined_details(request: ImageRequest):
    iris_result, hair_result, skin_tone_result = await asyncio.gather(
        get_iris_colour(request), get_hair_colour(request), get_skin_tone(request)
    )

    if "error" in iris_result:
        return iris_result
    if "error" in hair_result:
        return hair_result

    return {
        "eye_colour": iris_result["eye_colour"],
        "hair_colour": hair_result["hair_colour"],
        "skin_tone": skin_tone_result["skin_tone"]
    }

# To run the server, use: uvicorn <filename>:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
