import cv2
import os
from datetime import datetime
import requests
import base64
import json
import io
import os
import glob
import csv

def crop_roulette_area(image, top_left, bottom_right, frame_number, save=False, save_dir="../media/frame_cropped/"):
    """
    指定された座標で画像オブジェクトをクロップし、オプションで指定されたディレクトリに特定のフォーマットで保存する関数。
    
    :param image: クロップする画像オブジェクト
    :param top_left: クロップする領域の左上の座標 (x, y)
    :param bottom_right: クロップする領域の右下の座標 (x, y)
    :param frame_number: フレーム番号
    :param save: 画像を保存するかどうかのフラグ
    :param save_dir: 画像を保存するディレクトリのパス
    :return: クロップされた画像
    """
    
    # 画像をクロップする
    cropped_image = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    
    if save:
        # 保存するファイル名を生成する
        new_name = f"roulette_area_{frame_number}.jpg"
        os.makedirs(save_dir, exist_ok=True)  # ディレクトリがなければ作成
        save_path = os.path.join(save_dir, new_name)
        
        # クロップした画像を保存する
        cv2.imwrite(save_path, cropped_image)
        print(f"保存しました: {save_path}")
    
    return cropped_image

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
        return None

def crop_and_save_image(image_path, vertices, number, save_dir='../media/number_cropped/'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image from path: {image_path}")

    x_coords = [vertex[0] for vertex in vertices]
    y_coords = [vertex[1] for vertex in vertices]
    x1, x2 = min(x_coords), max(x_coords)
    y1, y2 = min(y_coords), max(y_coords)
    cropped_image = image[y1:y2, x1:x2]

    save_path = os.path.join(save_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}_{number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    cv2.imwrite(save_path, cropped_image)

def find_numbers_in_image(image_path, numbers_to_find, secrets_path='../../secrets/secrets.txt'):
    google_cloud_vision_api_url, api_key = load_api_settings(secrets_path)
    api_url = f"{google_cloud_vision_api_url}{api_key}"

    img_base64 = img_to_base64(image_path)
    result = request_cloud_vison_api(img_base64, api_url)

    if result is None:
        print("APIリクエストに失敗しました。")
        return [], []

    found_numbers = []
    found_vertices = []
    center_coordinates = []

    for number_to_find in numbers_to_find:
        closest_area_diff = float('inf')
        closest_vertices = None

        if "textAnnotations" in result["responses"][0]:
            for text_annotation in result["responses"][0]["textAnnotations"][1:]:
                detected_text = text_annotation["description"]
                if detected_text == number_to_find:
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
                x_center = (min(x_coords) + max(x_coords)) / 2
                y_center = (min(y_coords) + max(y_coords)) / 2
                center_coordinates.append((x_center, y_center))
                crop_and_save_image(image_path, closest_vertices, number_to_find)
                if number_to_find not in found_numbers:
                    found_numbers.append(number_to_find)
                    found_vertices.append(closest_vertices)
                print(center_coordinates)

    return found_numbers, center_coordinates

def save_roulette_data(video_name, found_numbers_data, execution_time, output_dir='../media/csv/'):
    """
    ルーレットの分析結果をCSVファイルに保存する関数。
    
    :param video_name: 分析された動画の名前
    :param found_numbers_data: 分析結果のデータ。各要素は (フレーム数, 発見された数字のリスト, 座標のリスト) のタプル。
    :param execution_time: 実行時刻
    :param output_dir: CSVを保存するディレクトリのパス
    """
    os.makedirs(output_dir, exist_ok=True)  # ディレクトリがなければ作成
    file_path = os.path.join(output_dir, f"analyze_roulette_data_{video_name}_{execution_time}.csv")
    
    with open(file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['フレーム数', '発見された数字', '座標'])
        
        for frame_number, numbers, vertices in found_numbers_data:
            # 座標リストを文字列に変換
            vertices_str = '; '.join([f"({x}, {y})" for x, y in vertices])
            # CSVに書き込み
            writer.writerow([frame_number, ', '.join(numbers), vertices_str])
    
    print(f"CSVファイルを保存しました: {file_path}")