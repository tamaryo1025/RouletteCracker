import requests
import base64
import json
import cv2
import os
from datetime import datetime

def load_api_settings(secrets_path='../secrets/secrets.txt'):
    with open(secrets_path) as f:
        lines = f.readlines()
        google_cloud_vision_api_url = lines[0].strip().split(" = ")[1]
        api_key = lines[1].strip().split(" = ")[1]
    return google_cloud_vision_api_url, api_key

def crop_image(image_path, top_left, bottom_right):
    image = cv2.imread(image_path)
    cropped_image = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    return cropped_image

def img_to_base64(filepath):
    with open(filepath, 'rb') as img:
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
    start_time = datetime.now()  # 時間計測開始
    res = requests.post(api_url, data=req_body)
    end_time = datetime.now()  # 時間計測終了
    duration = (end_time - start_time).total_seconds()  # 処理時間を秒単位で計算
    print(f"Vision APIの応答時間: {duration}秒")  # 処理時間を出力
    return res.json()

def process_image(image_path, top_left, bottom_right, secrets_path='../secrets/secrets.txt'):
    # 関数の属性を使ってディレクトリパスを保存・再利用
    if not hasattr(process_image, 'cropped_image_dir') or not hasattr(process_image, 'visionapi_output_dir'):
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        process_image.cropped_image_dir = f'../media/cropped/cropped_{current_time}/'
        process_image.visionapi_output_dir = f'../media/visionapi/visionapi_{current_time}/'
        os.makedirs(process_image.cropped_image_dir, exist_ok=True)
        os.makedirs(process_image.visionapi_output_dir, exist_ok=True)

    google_cloud_vision_api_url, api_key = load_api_settings(secrets_path)
    api_url = f"{google_cloud_vision_api_url}{api_key}"

    original_image_name = os.path.basename(image_path).split('.')[0]
    cropped_image_path = f"{process_image.cropped_image_dir}cropped_{original_image_name}.jpg"
    cropped_image = crop_image(image_path, top_left, bottom_right)
    cv2.imwrite(cropped_image_path, cropped_image)

    img_base64 = img_to_base64(cropped_image_path)
    result = request_cloud_vison_api(img_base64, api_url)

    if "textAnnotations" in result["responses"][0]:
        text_r = result["responses"][0]["textAnnotations"][1]["description"]
        print(text_r)
    else:
        text_r = "読み取り失敗"
        print("読み取り失敗")

    output_file_path = f"{process_image.visionapi_output_dir}recognize_list.txt"
    with open(output_file_path, 'a') as file:
        file.write(f'{original_image_name}: {text_r}\n')

if __name__ == "__main__":
    # このファイルが直接実行された場合のコード
    pass