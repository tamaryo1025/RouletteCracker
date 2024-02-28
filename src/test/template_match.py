import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# ディレクトリの存在を確認し、なければ作成
output_dir = "./images/template"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 類似度を格納するためのリスト
similarities = []

# テンプレート画像の読み込み
template = cv2.imread('./images/template_test.jpg', cv2.IMREAD_GRAYSCALE)

# フレーム1から400までの画像を処理
for frame_number in range(1, 401):
    # 画像の読み込み
    frame_path = f"../../media/frame/RouletteVideo_20240224/frame_{frame_number}.jpg"
    frame = cv2.imread(frame_path)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 画像が存在しない場合はスキップ
    if frame is None:
        print(f"フレーム {frame_number} を読み込めませんでした。")
        continue
    
    # 画像の切り取り
    cropped_frame = frame[560:640, 1170:1360]
    
    # 切り取った画像の保存
    cropped_frame_path = f"{output_dir}/cropped_template_{frame_number}.jpg"
    cv2.imwrite(cropped_frame_path, cropped_frame)
    
    # 類似度の計算
    similarity = cv2.matchTemplate(cropped_frame, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(similarity)
    similarities.append((frame_number, max_val))

# 最も類似度が高かったフレーム数の出力
most_similar_frame = max(similarities, key=lambda item: item[1])[0]
print(f"最も類似度が高かったフレーム数: {most_similar_frame}")

# 棒グラフの作成
frame_numbers, similarity_values = zip(*similarities)
plt.bar(frame_numbers, similarity_values)
plt.xlabel('フレーム数')
plt.ylabel('類似度')
plt.title('フレームごとの類似度')
plt.show()

import matplotlib.pyplot as plt

# 類似度の高い上位5つのフレームを取得
top_similar_frames = sorted(similarities, key=lambda item: item[1], reverse=True)[:5]

#  上位5つのフレームを画像として表示
for i, (frame_number, similarity) in enumerate(top_similar_frames, start=1):
    frame_path = f"../../media/frame/RouletteVideo_20240224/frame_{frame_number}.jpg"
    frame = cv2.imread(frame_path)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # OpenCVはBGRで読み込むので、RGBに変換
    plt.subplot(1,  5, i)  #  1行5列のサブプロットを作成し、現在のフレームを表示
    plt.imshow(frame)
    plt.title(f"フレーム番号: {frame_number}, 類似度: {similarity}")
    plt.axis('off')  # 軸を非表示にする

plt.show()
