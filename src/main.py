# import os
# import cv2
# import numpy as np
# from datetime import datetime
# from util import crop_roulette_area,find_numbers_in_image,save_roulette_data,save_roulette_data_streaming
# import time

# def load_images(background_image_path, mask_image_path):
#     """
#     背景画像とマスク画像を読み込む
#     """
#     background_image = cv2.imread(background_image_path)
#     mask_image = cv2.imread(mask_image_path, 0)
#     return background_image, mask_image

# def preprocess_images(background_image, mask_image):
#     """
#     画像を前処理する。マスク画像を二値化し、背景画像をグレースケールに変換してマスクを適用する。
#     """
#     _, mask = cv2.threshold(mask_image, 1, 255, cv2.THRESH_BINARY)
#     gray_background = cv2.cvtColor(background_image, cv2.COLOR_BGR2GRAY)
#     gray_background = cv2.bitwise_and(gray_background, gray_background, mask=mask)
#     return gray_background, mask

# def setup_video_io(input_video_path, output_video_path):
#     """
#     入力動画と出力動画の設定を行う。
#     """
#     cap = cv2.VideoCapture(input_video_path)
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
#     return cap, out, fps, frame_width, frame_height

# def detect_movement(gray_background, target_image, mask):
#     """
#     動きを検出する。背景と現在のフレームの差分を取り、変化があった部分を検出する。
#     """
#     gray_target = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
#     gray_target = cv2.bitwise_and(gray_target, gray_target, mask=mask)
#     difference = cv2.absdiff(gray_background, gray_target)
#     _, thresh = cv2.threshold(difference, 30, 255, cv2.THRESH_BINARY)
#     contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     return contours

# def find_largest_contour(contours):
#     """
#     最大の輪郭を見つける。
#     """
#     max_area = 0
#     max_contour = None
#     for contour in contours:
#         area = cv2.contourArea(contour)
#         if area > max_area:
#             max_area = area
#             max_contour = contour
#     return max_contour

# def calculate_speed(last_enter_frame, frame_number, fps, diameter_cm):
#     """
#     速度を計算する。
#     """
#     circumference_cm = np.pi * diameter_cm
#     elapsed_frames = frame_number - last_enter_frame
#     speed = circumference_cm / (elapsed_frames / fps)
#     return speed

# def process_video(input_video_path, output_video_path, background_image_path, mask_image_path, rect_top_left, rect_bottom_right, diameter_cm):
#     background_image, mask_image = load_images(background_image_path, mask_image_path)
#     gray_background, mask = preprocess_images(background_image, mask_image)
#     cap, out, fps, frame_width, frame_height = setup_video_io(input_video_path, output_video_path)

#     last_enter_frame = None
#     latest_speed = 0
#     found_numbers_data = []  # 分析結果を保存するためのリストを初期化

#     numbers_to_find = ["10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36", "37"]

#     while True:
#         ret, target_image = cap.read()
#         if not ret:
#             break
#         frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
#         contours = detect_movement(gray_background, target_image, mask)
#         max_contour = find_largest_contour(contours)

#         if max_contour is not None:
#             x, y, w, h = cv2.boundingRect(max_contour)
#             cv2.rectangle(target_image, (x, y), (x + w, y + h), (255, 0, 0), 2)

#             center_x, center_y = x + w // 2, y + h // 2
#             if rect_top_left[0] <= center_x <= rect_bottom_right[0] and rect_top_left[1] <= center_y <= rect_bottom_right[1]:
#                 if last_enter_frame is None or frame_number - last_enter_frame >= 5:
#                     if last_enter_frame is not None:
#                         latest_speed = calculate_speed(last_enter_frame, frame_number, fps, diameter_cm)
#                     last_enter_frame = frame_number
                    
#                     cropped_image = crop_roulette_area(target_image, (820, 570), (1370, 640), frame_number, save=True, save_dir="../media/frame_cropped/")
#                     found_numbers, found_vertices, found_angles = find_numbers_in_image(f"../media/frame_cropped/roulette_area_{frame_number}.jpg", numbers_to_find, (820, 570), secrets_path='../secrets/secrets.txt')
#                     if found_numbers:
#                         found_numbers_data.append((frame_number, found_numbers, found_vertices, found_angles))  # 角度の情報を含める

#         # 速度テキストを画像に追加
#         speed_text = f"速度: {latest_speed:.2f} cm/s"
#         cv2.putText(target_image, speed_text, (10, frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
#         out.write(target_image)

#     # リソースの解放
#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()

#     # 分析結果をCSVファイルに保存
#     execution_time = datetime.now().strftime("%Y%m%d_%H%M%S")
#     save_roulette_data(input_video_name, found_numbers_data, execution_time)

# def process_video_streaming(background_image_path, mask_image_path, rect_top_left, rect_bottom_right, diameter_cm):
#     background_image, mask_image = load_images(background_image_path, mask_image_path)
#     gray_background, mask = preprocess_images(background_image, mask_image)
#     cap = cv2.VideoCapture('../media/videos/RouletteVideo_20240321.mov') # ウェブカメラからのストリーミングを開始

#     # fpsを取得
#     fps = cap.get(cv2.CAP_PROP_FPS)

#     # 実行時刻を取得（この実行セッション中に作成されるCSVファイル名に使用）
#     execution_time = datetime.now().strftime("%Y%m%d_%H%M%S")

#     last_enter_frame = None
#     latest_speed = 0
#     found_numbers_data = []

#     numbers_to_find = ["10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36", "37"]

#     while True:
#         ret, target_image = cap.read()
#         if not ret:
#             break
#         frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
#         contours = detect_movement(gray_background, target_image, mask)
#         max_contour = find_largest_contour(contours)

#         if max_contour is not None:
#                     x, y, w, h = cv2.boundingRect(max_contour)
#                     cv2.rectangle(target_image, (x, y), (x + w, y + h), (255, 0, 0), 2)

#                     center_x, center_y = x + w // 2, y + h // 2
#                     if rect_top_left[0] <= center_x <= rect_bottom_right[0] and rect_top_left[1] <= center_y <= rect_bottom_right[1]:
#                         if last_enter_frame is None or frame_number - last_enter_frame >= 5:
#                             if last_enter_frame is not None:
#                                 latest_speed = calculate_speed(last_enter_frame, frame_number, fps, diameter_cm)
#                             last_enter_frame = frame_number
                            
#                             cropped_image = crop_roulette_area(target_image, (820, 570), (1370, 640), frame_number, save=True, save_dir="../media/frame_cropped/")
#                             found_numbers, found_vertices, found_angles = find_numbers_in_image(f"../media/frame_cropped/roulette_area_{frame_number}.jpg", numbers_to_find, (820, 570), secrets_path='../secrets/secrets.txt')
#                             if found_numbers:
#                                 # found_numbers_data.append((frame_number, found_numbers, found_vertices, found_angles))
#                                 # execution_time = datetime.now().strftime("%Y%m%d_%H%M%S")  # 実行時刻を取得
#                                 save_roulette_data_streaming(input_video_name, frame_number, found_numbers, found_vertices, found_angles,frame_image=target_image)

#     cap.release()
#     cv2.destroyAllWindows()

import os
import cv2
import numpy as np
from datetime import datetime
from util import crop_roulette_area, find_numbers_in_image, save_roulette_data, save_roulette_data_streaming, calculate_angle_with_tilt
import time

def load_images(background_image_path, mask_image_path):
    background_image = cv2.imread(background_image_path)
    mask_image = cv2.imread(mask_image_path, 0)
    return background_image, mask_image

def preprocess_images(background_image, mask_image):
    _, mask = cv2.threshold(mask_image, 1, 255, cv2.THRESH_BINARY)
    gray_background = cv2.cvtColor(background_image, cv2.COLOR_BGR2GRAY)
    gray_background = cv2.bitwise_and(gray_background, gray_background, mask=mask)
    return gray_background, mask

def setup_video_io(input_video_path, output_video_path):
    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    return cap, out, fps, frame_width, frame_height

def detect_and_draw_contours(gray_background, target_image, mask):
    gray_target = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
    gray_target = cv2.bitwise_and(gray_target, gray_target, mask=mask)
    difference = cv2.absdiff(gray_background, gray_target)
    _, thresh = cv2.threshold(difference, 30, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = find_largest_contour(contours)
    if max_contour is not None:
        x, y, w, h = cv2.boundingRect(max_contour)
        cv2.rectangle(target_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        return (x + w // 2, y + h // 2), max_contour  # Return center coordinates and max_contour
    return None, None

def find_largest_contour(contours):
    max_area = 0
    max_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour
    return max_contour

def calculate_speed(last_enter_frame, frame_number, fps, diameter_cm):
    circumference_cm = np.pi * diameter_cm
    elapsed_frames = frame_number - last_enter_frame
    speed = circumference_cm / (elapsed_frames / fps)
    return speed

def process_frame(target_image, gray_background, mask, frame_number, last_enter_frame, fps, diameter_cm, numbers_to_find):
    center, max_contour = detect_and_draw_contours(gray_background, target_image, mask)
    latest_speed = 0 
    if center:
        center_x, center_y = center
        if last_enter_frame is None or frame_number - last_enter_frame >= 5:
            if last_enter_frame is not None:
                latest_speed = calculate_speed(last_enter_frame, frame_number, fps, diameter_cm)
            last_enter_frame = frame_number
            cropped_image = crop_roulette_area(target_image, (820, 570), (1370, 640), frame_number, save=True, save_dir="../media/frame_cropped/")
            found_numbers, found_vertices, found_angles = find_numbers_in_image(f"../media/frame_cropped/roulette_area_{frame_number}.jpg", numbers_to_find, (820, 570), secrets_path='../secrets/secrets.txt')
            if found_numbers:
                save_roulette_data_streaming(input_video_name, frame_number, found_numbers, found_vertices, found_angles, frame_image=target_image)
        angle = calculate_angle_with_tilt(center)
        print(f"フレーム {frame_number}: 中心座標 ({center_x}, {center_y}) の角度 = {angle:.2f}度")
    return last_enter_frame, latest_speed

def process_video_streaming(background_image_path, mask_image_path, rect_top_left, rect_bottom_right, diameter_cm):
    background_image, mask_image = load_images(background_image_path, mask_image_path)
    gray_background, mask = preprocess_images(background_image, mask_image)
    cap = cv2.VideoCapture('../media/videos/RouletteVideo_20240321.mov')

    fps = cap.get(cv2.CAP_PROP_FPS)
    execution_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    last_enter_frame = None
    latest_speed = 0
    numbers_to_find = ["10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36", "37"]

    while True:
        ret, target_image = cap.read()
        if not ret:
            break
        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        last_enter_frame, latest_speed = process_frame(target_image, gray_background, mask, frame_number, last_enter_frame, fps, diameter_cm, numbers_to_find)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    input_video_path = '../media/videos/RouletteVideo_20240321.mov'
    input_video_name = os.path.splitext(os.path.basename(input_video_path))[0]
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_directory = f'../media/result/result_{current_time}/'
    os.makedirs(output_directory, exist_ok=True)
    output_video_path = f'{output_directory}detect_{input_video_name}.mp4'
    background_image_path = '../media/frame/RouletteVideo_20240224/frame_0.jpg'
    mask_image_path = '../media/frame/RouletteVideo_20240224/frame_mask_2.jpg'
    rect_top_left = (1670, 550)
    rect_bottom_right = (1800, 650)
    diameter_cm = 50

    process_video_streaming(background_image_path, mask_image_path, rect_top_left, rect_bottom_right, diameter_cm)