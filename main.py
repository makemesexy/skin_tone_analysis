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
from openai import AzureOpenAI

API_KEY = "f85804b527a943f197ee55b7df623528"
API_VERSION = "2024-02-15-preview"
AZURE_ENDPOINT = "https://embedding-model12.openai.azure.com"

# Initialize the Azure OpenAI client
client = AzureOpenAI(
    api_key=API_KEY,
    api_version=API_VERSION,
    azure_endpoint=AZURE_ENDPOINT
)

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

# Eye color detection
async def get_iris_colour(image: np.ndarray):
    frame = cv2.flip(image, 1)
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

            return final_hex_code
        else:
            raise HTTPException(status_code=400, detail="No face landmarks detected.")

# Hair color detection
async def get_hair_colour(image: np.ndarray):
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

        return dominant_color_hex

# Skin tone detection
async def get_skin_tone(image_url: str):
    result = stone.process(image_url, image_type="color", return_report_image=True)
    skin_tone = result["faces"][0]["skin_tone"]
    return skin_tone

# Color palette generation
def get_colour_palette(eye_colour, hair_colour, skin_tone):
    messages = [
        {
            "role": "system",
            "content": "You are an AI assistant that creates a color palette for user based on eye color, hair color, and skin tone, give to the point answer"
        },
        {
            "role": "user",
            "content": f"My eye color is {eye_colour}, my hair color is {hair_colour}, and my skin tone is {skin_tone}. Can you suggest a suitable color palette like autumn,summer,spring,winter,pls just give me short to the point answer?"
        }
    ]
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # Ensure you are using the correct model name
            messages=messages,
            max_tokens=100  # You can adjust this value as needed
        )
        palette = response.choices[0].message.content.lower()

        if "autumn" in palette:
            return autumn_palette, "autumn"
        elif "summer" in palette:
            return summer_palette, "summer"
        elif "spring" in palette:
            return spring_palette, "spring"
        elif "winter" in palette:
            return winter_palette, "winter"
        else:
            return default_palette, "default"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get response from OpenAI. Error: {e}")

# Color palettes
autumn_palette = [
    "#FF8C00",  # Dark Orange
    "#8B0000",  # Dark Red
    "#B8860B",  # Dark Goldenrod
    "#6A5ACD",  # Slate Blue
    "#DAA520"   # Goldenrod
]

summer_palette = [
    "#FF6347",  # Tomato
    "#00CED1",  # Dark Turquoise
    "#FFD700",  # Gold
    "#FF69B4",  # Hot Pink
    "#87CEFA"   # Light Sky Blue
]

spring_palette = [
    "#00FF7F",  # Spring Green
    "#FFB6C1",  # Light Pink
    "#FFD700",  # Gold
    "#8A2BE2",  # Blue Violet
    "#98FB98"   # Pale Green
]

winter_palette = [
    "#4682B4",  # Steel Blue
    "#1E90FF",  # Dodger Blue
    "#FFFAFA",  # Snow White
    "#D3D3D3",  # Light Grey
    "#000080"   # Navy
]

default_palette = [
    "#FF0000",  # red
    "#008000",  # green
    "#FFFFFF",  # White
    "#000000",  # Black
    "#000080"   # Navy
]

class ImageInput(BaseModel):
    image_url: str

@app.post("/detect_features")
async def detect_features(image_input: ImageInput):
    image_url = image_input.image_url
    
    try:
        image = await download_image(image_url)
        
        eye_colour_hex = await get_iris_colour(image)
        hair_colour_hex = await get_hair_colour(image)
        skin_tone = await get_skin_tone(image_url)
        colour_palette, season = get_colour_palette(eye_colour_hex, hair_colour_hex, skin_tone)

        response = {
            "eye_colour": eye_colour_hex,
            "hair_colour": hair_colour_hex,
            "skin_tone": skin_tone,
            "colour_palette": colour_palette,
            "season": season
        }
        
        return JSONResponse(content=response)
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)