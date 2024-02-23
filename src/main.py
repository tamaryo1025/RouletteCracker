import cv2
import numpy as np

def process_video(input_video_path, output_video_path, background_image_path, mask_image_path, rect_top_left, rect_bottom_right, diameter_cm):
    # 背景画像とマスク画像の読み込み
    background_image = cv2.imread(background_image_path)
    mask_image = cv2.imread(mask_image_path, 0)

    # マスク画像を二値化（黒塗り部分が255、それ以外が0）
    _, mask = cv2.threshold(mask_image, 1, 255, cv2.THRESH_BINARY)

    # 背景画像をグレースケールに変換し、マスクを適用
    gray_background = cv2.cvtColor(background_image, cv2.COLOR_BGR2GRAY)
    gray_background = cv2.bitwise_and(gray_background, gray_background, mask=mask)

    # 入力動画を読み込む
    cap = cv2.VideoCapture(input_video_path)

    # 出力動画の設定（フレームレートとサイズ）
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    # 経過フレーム数と円周の長さで速度を判断
    circumference_cm = np.pi * diameter_cm  # 円周

    # 前回長方形内に入ったフレーム数
    last_enter_frame = None

    # 最新の速度
    latest_speed = 0

    while True:
        ret, target_image = cap.read()
        if not ret:
            break

        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))  # 現在のフレーム数

        gray_target = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
        gray_target = cv2.bitwise_and(gray_target, gray_target, mask=mask)

        difference = cv2.absdiff(gray_background, gray_target)
        _, thresh = cv2.threshold(difference, 30, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        max_area = 0
        max_contour = None
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                max_contour = contour

        if max_contour is not None:
            x, y, w, h = cv2.boundingRect(max_contour)
            # 球に青い枠をつける
            cv2.rectangle(target_image, (x, y), (x + w, y + h), (255, 0, 0), 2)

            center_x, center_y = x + w // 2, y + h // 2
            if rect_top_left[0] <= center_x <= rect_bottom_right[0] and rect_top_left[1] <= center_y <= rect_bottom_right[1]:
                if last_enter_frame is None or frame_number - last_enter_frame >= 5:
                    if last_enter_frame is not None:
                        elapsed_frames = frame_number - last_enter_frame
                        latest_speed = circumference_cm / (elapsed_frames / fps)
                    last_enter_frame = frame_number

        # 速度テキストを画像に追加
        speed_text = f"速度: {latest_speed:.2f} cm/s"
        cv2.putText(target_image, speed_text, (10, frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        out.write(target_image)

    # リソースの解放
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# 使用例
input_video_path = '../media/videos/ball2.mp4'
output_video_path = '../media/result2/detected_ball_video.mp4'
background_image_path = '../media/images/frame_279.jpg'
mask_image_path = '../media/images/frame_mask.jpg'
rect_top_left = (350, 70)
rect_bottom_right = (400, 100)
diameter_cm = 50

process_video(input_video_path, output_video_path, background_image_path, mask_image_path, rect_top_left, rect_bottom_right, diameter_cm)