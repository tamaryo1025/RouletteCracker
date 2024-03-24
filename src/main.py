import os
import cv2
import numpy as np
from datetime import datetime
from util import *
import time

# グローバル変数で座標を定義
# CROP_DETECT_NUMBER_AREA_TOP_LEFT = (820, 570)
# CROP_DETECT_NUMBER_AREA_BOTTOM_RIGHT = (1370, 640)
CROP_DETECT_NUMBER_AREA_TOP_LEFT = (680, 2050)
CROP_DETECT_NUMBER_AREA_BOTTOM_RIGHT = (1500, 2400)
# CROP_DETECT_BALL_AREA_TOP_LEFT = (1670, 550)
# CROP_DETECT_BALL_AREA_BOTTOM_RIGHT = (1800, 650)
CROP_DETECT_BALL_AREA_TOP_LEFT = (760, 470)
CROP_DETECT_BALL_AREA_BOTTOM_RIGHT = (870, 530)
# ROULETTE_CENTER = (1080, 760)
ROULETTE_CENTER = (1080, 2730)

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

def process_frame(target_image, gray_background, mask, frame_number, last_enter_frame, fps, diameter_cm, numbers_to_find, input_video_name):
    latest_speed = 0
    global CROP_DETECT_BALL_AREA_TOP_LEFT, CROP_DETECT_BALL_AREA_BOTTOM_RIGHT
    center, max_contour = detect_and_draw_contours(gray_background, target_image, mask)

    if center:
        center_x, center_y = center
        if CROP_DETECT_BALL_AREA_TOP_LEFT[0] <= center_x <= CROP_DETECT_BALL_AREA_BOTTOM_RIGHT[0] and CROP_DETECT_BALL_AREA_TOP_LEFT[1] <= center_y <= CROP_DETECT_BALL_AREA_BOTTOM_RIGHT[1]:
            if last_enter_frame is None or frame_number - last_enter_frame >= 5:

                # ターゲット画像を保存するディレクトリとファイル名を指定
                save_dir = "../media/frame_ball_in/"
                os.makedirs(save_dir, exist_ok=True)  # ディレクトリがなければ作成
                current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(save_dir, f"frame_{frame_number}_{current_time}.jpg")
                # ターゲット画像を保存
                cv2.imwrite(save_path, target_image)

                if last_enter_frame is not None:
                    latest_speed = calculate_speed(last_enter_frame, frame_number, fps, diameter_cm)
                last_enter_frame = frame_number

                #画像は上から見た画角にreshape
                cropped_image = crop_roulette_area(reshape_frame(target_image), CROP_DETECT_NUMBER_AREA_TOP_LEFT, CROP_DETECT_NUMBER_AREA_BOTTOM_RIGHT, frame_number, save=True, save_dir="../media/frame_cropped/")
                found_numbers, found_vertices, found_angles = find_numbers_in_image(f"../media/frame_cropped/roulette_area_{frame_number}.jpg", numbers_to_find, CROP_DETECT_NUMBER_AREA_TOP_LEFT,ROULETTE_CENTER, secrets_path='../secrets/secrets.txt')
                if found_numbers:
                    save_roulette_data_streaming(input_video_name, frame_number, found_numbers, found_vertices, found_angles, frame_image=target_image)
        angle = calculate_angle_with_tilt(center,ROULETTE_CENTER)
        print(f"フレーム {frame_number}: 中心座標 ({center_x}, {center_y}) の角度 = {angle:.2f}度")
    return last_enter_frame, latest_speed

def process_video_streaming(background_image_path, mask_image_path, input_video_path, diameter_cm):
    background_image, mask_image = load_images(background_image_path, mask_image_path)
    gray_background, mask = preprocess_images(background_image, mask_image)
    video_name = os.path.basename(input_video_path)
    
    # 現在の時刻を使用して出力ディレクトリと出力ファイルパスを設定
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_directory = f'../media/result/result_{current_time}/'
    os.makedirs(output_directory, exist_ok=True)
    output_video_path = f'{output_directory}detect_{os.path.splitext(os.path.basename(input_video_path))[0]}.mp4'
    
    # setup_video_io 関数を使用して動画の入力と出力を設定
    cap, out, fps, frame_width, frame_height = setup_video_io(input_video_path, output_video_path)

    last_enter_frame = None
    latest_speed = 0
    numbers_to_find = ["10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36", "37"]

    while True:
        ret, target_image = cap.read()
        if not ret:
            break
        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        last_enter_frame, latest_speed = process_frame(target_image, gray_background, mask, frame_number, last_enter_frame, fps, diameter_cm, numbers_to_find, video_name)
        # 処理されたフレームを出力ビデオに書き込む
        out.write(target_image)
    cap.release()
    out.release()  # 出力ビデオを正しく閉じる
    cv2.destroyAllWindows()

if __name__ == "__main__":
    input_video_path = '../media/videos/RouletteVideo_20240225.mov'
    background_image_path = '../media/frame/RouletteVideo_20240224/frame_0.jpg'
    mask_image_path = '../media/frame/RouletteVideo_20240224/frame_mask_2.jpg'
    diameter_cm = 50

    process_video_streaming(background_image_path, mask_image_path, input_video_path, diameter_cm)