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
import numpy as np

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

def modify_vertices_center(vertices,adjusted_vertices):
    
    x_coords = [vertex[0] for vertex in vertices]
    y_coords = [vertex[1] for vertex in vertices]
    x1, x2 = min(x_coords), max(x_coords)
    y1, y2 = min(y_coords), max(y_coords)

    center_x = x1 + (x2 - x1) / 2 + adjusted_vertices[0]
    center_y = y1 + (y2 - y1) / 2 + adjusted_vertices[1]
    
    return  (center_x, center_y)

def crop_and_save_image(image_path, vertices,adjusted_vertices, number, save_dir='../media/number_cropped/'):
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

    # 保存する画像名に切り抜いた画像の座標を含める
    save_path = os.path.join(save_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}_{number}_({modify_vertices_center(vertices,adjusted_vertices)[0]},{modify_vertices_center(vertices,adjusted_vertices)[1]})_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    # cv2.imwrite(save_path, cropped_image)

def find_numbers_in_image(image_path, numbers_to_find, adjusted_vertices ,secrets_path='../secrets/secrets.txt'):
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
    found_angles = []

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
                crop_and_save_image(image_path, closest_vertices,adjusted_vertices, number_to_find)
                if number_to_find not in found_numbers:
                    found_numbers.append(number_to_find)
                    found_vertices.append(closest_vertices)
                center_coordinates.append(modify_vertices_center(closest_vertices,adjusted_vertices))
                # 角度を計算してリストに追加
                angle = calculate_angle_with_tilt(modify_vertices_center(closest_vertices,adjusted_vertices))
                found_angles.append(angle)
                print(center_coordinates)

    return found_numbers, center_coordinates, found_angles

def save_roulette_data(video_name, found_numbers_data, execution_time, output_dir='../media/csv/'):
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"analyze_roulette_data_{video_name}_{execution_time}.csv")
    
    with open(file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['フレーム数', '発見された数字', '座標', '角度'])  # 角度の列を追加
        
        for frame_number, numbers, vertices, angles in found_numbers_data:  # 角度の情報を含める
            vertices_str = '; '.join([f"({x}, {y})" for x, y in vertices])
            angles_str = ', '.join([f"{angle:.2f}" for angle in angles])  # 角度を文字列に変換
            writer.writerow([frame_number, ', '.join(numbers), vertices_str, angles_str])  # 角度も書き込む
    
    print(f"CSVファイルを保存しました: {file_path}")

def calculate_tilt_angle(long_axis, short_axis):
    """
    楕円の長軸と短軸の長さから、真上から見た時の傾いた角度θを求める関数。
    
    :param long_axis: 楕円の長軸の長さ
    :param short_axis: 楕円の短軸の長さ
    :return: 傾いた角度θ（ラジアン単位）
    """
    return np.arccos(short_axis / long_axis)

def convert_coordinates(x, y, theta):
    """
    傾いたカメラ座標での（x、y）の座標を真上から見た場合の座標（X ,Y)に変換する関数。
    この時、カメラの中心と円の中心は70cmの距離があると仮定する。
    
    :param x: 傾いたカメラ座標でのx座標
    :param y: 傾いたカメラ座標でのy座標
    :param theta: 傾いた角度θ（ラジアン単位）
    :return: 真上から見た場合の座標（X ,Y)
    """
    # 中心との距離を保持しながら座標変換を行う
    distance = 70  # 中心との距離(cm)
    X = (x + distance * np.cos(theta)) * np.cos(theta) + (y + distance * np.sin(theta)) * np.sin(theta)
    Y = -(x + distance * np.cos(theta)) * np.sin(theta) + (y + distance * np.sin(theta)) * np.cos(theta)
    return X, Y

def calculate_angle_with_tilt(cordinate, tilt_angle=74):
    """
    座標（x,y）と傾きが与えられた時の、基準ベクトルとの角度を算出する関数。マイナスの角度も考慮する。
    """
    
    # 中心座標を引いて調整
    center_x = 1080
    center_y = 740
    adjusted_x = cordinate[0] - center_x
    adjusted_y = cordinate[1] - center_y
    
    # 傾きを考慮して座標変換
    theta = np.radians(tilt_angle)  # 度数法からラジアンに変換
    X = adjusted_x
    Y = + adjusted_y / np.cos(theta)
    
    # 基準ベクトルとの角度を計算
    v1 = (0, -873)  # 基準ベクトル
    v2 = (X, Y)  # 変換後の座標をベクトルとして
    
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    angle_rad = np.arccos(dot_product / (norm_v1 * norm_v2))
    angle_deg = np.degrees(angle_rad)
    
    # X座標が負の場合、角度をマイナスにする
    if adjusted_x < 0:
        angle_deg = -angle_deg
    
    return angle_deg