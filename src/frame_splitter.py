import cv2
import os

def convert_to_mp4(input_path, output_path):
    # 入力動画を読み込む
    cap = cv2.VideoCapture(input_path)
    # 出力フォーマット(MP4)とフレームレート、サイズを設定
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 動画をフレームごとに読み込み、出力動画に書き込む
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
    
    # リソースを解放
    cap.release()
    out.release()

def split_video_into_frames(video_path):
    # ファイルの拡張子をチェック
    ext = os.path.splitext(video_path)[1]
    if ext != '.mp4':
        # MP4形式以外の場合は変換
        print(f"{ext}形式の動画をMP4に変換します。")
        converted_path = os.path.splitext(video_path)[0] + '.mp4'
        convert_to_mp4(video_path, converted_path)
        video_path = converted_path
    
    cap = cv2.VideoCapture(video_path)
    frame_number = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        output_directory = f"../media/frame/{os.path.splitext(os.path.basename(video_path))[0]}/"
        os.makedirs(output_directory, exist_ok=True)

        frame_path = os.path.join(output_directory, f"frame_{frame_number}.jpg")
        cv2.imwrite(frame_path, frame)

        frame_number += 1

    cap.release()

# 使用例
if __name__ == "__main__":
    video_path = '../media/videos/RouletteVideo_20240224.mp4'
    split_video_into_frames(video_path)