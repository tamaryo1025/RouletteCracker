import requests
import base64
import json
import io
import os
import cv2
from datetime import datetime
import concurrent.futures
import time
from tqdm import tqdm

def load_api_settings(secrets_path='../../secrets/secrets.txt'):
    with open(secrets_path) as f:
        lines = f.readlines()
        google_cloud_vision_api_url = lines[0].strip().split(" = ")[1]
        api_key = lines[1].strip().split(" = ")[1]
    return google_cloud_vision_api_url, api_key

def img_to_base64(image_path):
    with io.open(image_path, 'rb') as img:
        img_byte = img.read()
    return base64.b64encode(img_byte)

def request_cloud_vison_api(image_base64, api_url):
    req_body = json.dumps({
        'requests': [{
            'image': {
                'content': image_base64.decode('utf-8')
            },
            'features': [{
                'type': 'TEXT_DETECTION',
                'maxResults': 10,
            }]
        }]
    })
    try:
        res = requests.post(api_url, data=req_body, timeout=0.9)
        return res.json()
    except requests.exceptions.Timeout:
        # Vision APIの応答時間が0.8秒を超えた場合の出力を削除
        return None

def crop_and_save_image(image_path, vertices, number, save_dir='./images/'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    image = cv2.imread(image_path)
    x_coords = [vertex[0] for vertex in vertices]
    y_coords = [vertex[1] for vertex in vertices]
    x1, x2 = min(x_coords), max(x_coords)
    y1, y2 = min(y_coords), max(y_coords)
    cropped_image = image[y1:y2, x1:x2]
    save_path = os.path.join(save_dir, f"cropped_{number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    # cv2.imwrite(save_path, cropped_image)
    # 画像保存の出力を削除

def find_number_in_image(image_path, number_to_find, secrets_path='../../secrets/secrets.txt'):
    google_cloud_vision_api_url, api_key = load_api_settings(secrets_path)
    api_url = f"{google_cloud_vision_api_url}{api_key}"

    img_base64 = img_to_base64(image_path)
    result = request_cloud_vison_api(img_base64, api_url)

    if result is None:
        return

    found = False
    closest_area_diff = float('inf')
    closest_vertices = None

    if "textAnnotations" in result["responses"][0]:
        for text_annotation in result["responses"][0]["textAnnotations"][1:]:
            if number_to_find in text_annotation["description"]:
                vertices = [(vertex.get("x", 0), vertex.get("y", 0)) for vertex in text_annotation["boundingPoly"]["vertices"]]
                x_coords = [vertex[0] for vertex in vertices]
                y_coords = [vertex[1] for vertex in vertices]
                area = (max(x_coords) - min(x_coords)) * (max(y_coords) - min(y_coords))
                
                if area <= 5000:
                    area_diff = abs(2000 - area)
                    if area_diff < closest_area_diff:
                        closest_area_diff = area_diff
                        closest_vertices = vertices

    if closest_vertices:
        crop_and_save_image(image_path, closest_vertices, number_to_find)
        found = True

    # 数字が見つからなかった場合の出力を削除

def main():
    image_path = './images/roulette_only.png'
    numbers_to_find = [str(number) for number in range(10, 38)]  # 1から37までの数字のリスト

    start_time = time.time()

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(find_number_in_image, image_path, number) for number in numbers_to_find]
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(numbers_to_find), desc="Processing"):
          pass

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Completed all processes in {elapsed_time:.2f} seconds.")

if __name__ == "__main__":
    main()