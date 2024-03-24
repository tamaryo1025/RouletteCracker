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
from PIL import Image

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

def crop_roulette_area_with_angle(image, top_left, bottom_right, frame_number, angle=0, save=False, save_dir="../media/frame_cropped/"):
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
    cv2.imwrite(save_path, cropped_image)

def find_numbers_in_image(image_path, numbers_to_find, adjusted_vertices ,center, secrets_path='../secrets/secrets.txt'):
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
                    
                    if area <= 100000000:
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
                angle = calculate_angle_with_tilt(modify_vertices_center(closest_vertices,adjusted_vertices),center)
                print(f"modify_center:{modify_vertices_center(closest_vertices,adjusted_vertices)}")
                print(f"roulette_center:{center}")
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

# def save_roulette_data_streaming(video_name, frame_number, found_numbers, found_vertices, found_angles, execution_time, output_dir='../media/csv/'):
#     os.makedirs(output_dir, exist_ok=True)
#     file_path = os.path.join(output_dir, f"analyze_roulette_data_{video_name}_{execution_time}.csv")
    
#     with open(file_path, mode='a', newline='', encoding='utf-8') as file:
#         writer = csv.writer(file)
#         if os.stat(file_path).st_size == 0:  # ファイルが空の場合、ヘッダーを追加
#             writer.writerow(['フレーム数', '発見された数字', '座標', '角度'])
        
#         vertices_str = '; '.join([f"({x}, {y})" for x, y in found_vertices])
#         angles_str = ', '.join([f"{angle:.2f}" for angle in found_angles])
#         writer.writerow([frame_number, ', '.join(found_numbers), vertices_str, angles_str])

# グローバル変数として前回書き込まれたフレーム数を初期化
last_written_frame = -1001  # 初期値は-1001として、最初の書き込みが必ず行われるようにする

def save_roulette_data_streaming(video_name, frame_number, found_numbers, found_vertices, found_angles,frame_image):
    global last_written_frame  # グローバル変数を関数内で使用する宣言
    
    # 出力ディレクトリを指定
    output_dir = '../media/csv/'
    
    # フレーム数が前回の書き込みから1000以上離れている場合、新しいファイル名を生成
    if frame_number - last_written_frame >= 1000 or last_written_frame == -1001:
        execution_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(output_dir, f"analyze_roulette_data_{video_name}_{execution_time}.csv")
        last_written_frame = frame_number  # 現在のフレーム数を記録
        # 新しいCSVファイルが作成された時に実行
        cropped_image = crop_roulette_area(frame_image, (0,0), (2160,1350), frame_number, save=False)

        #最終結果の読み取り
        detected_texts = find_number_in_specified_area(cropped_image, top_left=(1550, 960), bottom_right=(1590, 980), secrets_path='../secrets/secrets.txt')
        with open('result.txt', mode='a', encoding='utf-8') as file:
            file.write(detected_texts + '\n')  # 座標と角度は空で追記
    else:
        # 最後に書き込まれたファイルを探す
        list_of_files = glob.glob(f'{output_dir}analyze_roulette_data_{video_name}_*.csv')  # パターンにマッチするファイルのリストを取得
        if list_of_files:
            latest_file = max(list_of_files, key=os.path.getctime)  # 最新のファイルを取得
            file_path = latest_file
        else:
            # 予期せぬ理由でファイルが見つからない場合は新しいファイルを作成
            execution_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(output_dir, f"analyze_roulette_data_{video_name}_{execution_time}.csv")
    
    os.makedirs(output_dir, exist_ok=True)
    
    with open(file_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if os.stat(file_path).st_size == 0:  # ファイルが空の場合、ヘッダーを追加
            writer.writerow(['フレーム数', '発見された数字', '座標', '角度'])
        
        vertices_str = '; '.join([f"({x}, {y})" for x, y in found_vertices])
        angles_str = ', '.join([f"{angle:.2f}" for angle in found_angles])
        writer.writerow([frame_number, ', '.join(found_numbers), vertices_str, angles_str])

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
    """
    X = x
    Y = + y / np.cos(theta)
    return X, Y

def calculate_angle_with_tilt(cordinate, center, tilt_angle=74):
    """
    座標（x,y）と傾きが与えられた時の、基準ベクトルとの角度を算出する関数。マイナスの角度も考慮する。
    """
    
    # 中心座標を引いて調整
    # center_x = 1080
    # center_y = 740
    center_x = center[0]
    center_y = center[1]
    adjusted_x = cordinate[0] - center_x
    adjusted_y = cordinate[1] - center_y
    
    # # 傾きを考慮して座標変換
    # theta = np.radians(tilt_angle)  # 度数法からラジアンに変換
    # X = adjusted_x
    # Y = + adjusted_y / np.cos(theta)
    
    # 基準ベクトルとの角度を計算
    v1 = (0, -1)  # 基準ベクトル
    v2 = (adjusted_x, adjusted_y)  # 変換後の座標をベクトルとして
    
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    angle_rad = np.arccos(dot_product / (norm_v1 * norm_v2))
    angle_deg = np.degrees(angle_rad)
    
    # X座標が負の場合、角度をマイナスにする
    if adjusted_x < 0:
        angle_deg = -angle_deg
    
    return angle_deg

def find_number_in_specified_area(image, top_left=(1550, 960), bottom_right=(1590, 980), secrets_path='../secrets/secrets.txt'):
    """
    指定された座標で囲まれる長方形内にある数字を読み取る。
    
    :param image: 読み取りを行う画像オブジェクト
    :param top_left: 長方形の左上の座標 (x, y)
    :param bottom_right: 長方形の右下の座標 (x, y)
    :param secrets_path: APIキーが保存されているファイルのパス
    :return: 検出されたテキストのリスト
    """
    # 画像をクロップする
    cropped_image = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    
    # クロップされた画像をbase64エンコードする
    _, img_encoded = cv2.imencode('.jpg', cropped_image)
    img_base64 = base64.b64encode(img_encoded).decode('utf-8')
    
    # API設定を読み込む
    google_cloud_vision_api_url, api_key = load_api_settings(secrets_path)
    api_url = f"{google_cloud_vision_api_url}{api_key}"
    
    # Cloud Vision APIにリクエストを送信
    req_body = json.dumps({
        'requests': [{
            'image': {
                'content': img_base64
            },
            'features': [{
                'type': 'TEXT_DETECTION',
                'maxResults': 10,
            }]
        }]
    })
    
    try:
        res = requests.post(api_url, data=req_body, timeout=0.9)
        result = res.json()
    except requests.exceptions.Timeout:
        print("APIリクエストに失敗しました。")
        return []
    
    # 検出されたテキストを抽出
    detected_texts = []
    if "textAnnotations" in result["responses"][0]:
        for text_annotation in result["responses"][0]["textAnnotations"]:
            detected_text = text_annotation["description"]
            detected_texts.append(detected_text)
    
    if len(detected_texts)==0:
        return "none"
    
    return detected_texts[-1]

def reshape_frame(target_image, angle=74):

    image = target_image

    # 画像のサイズを取得
    height, width = image.shape[:2]

    # 変換後の画像サイズを計算するために、四隅の座標を変換
    corners = [(0, 0), (width-1, 0), (0, height-1), (width-1, height-1)]
    converted_corners = [convert_coordinates(x, y, np.radians(angle)) for x, y in corners]
    min_x = min([x for x, y in converted_corners])
    max_x = max([x for x, y in converted_corners])
    min_y = min([y for x, y in converted_corners])
    max_y = max([y for x, y in converted_corners])

    # 新しい画像のサイズを設定
    new_width = int(max_x - min_x)
    new_height = int(max_y - min_y)

    # 元の画像を新しいサイズにリサイズ
    resized_image = cv2.resize(image, (new_width, new_height))

    return resized_image