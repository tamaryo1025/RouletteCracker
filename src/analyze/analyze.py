import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import japanize_matplotlib

# ベースディレクトリを設定
base_dir = '../../media/visionapi/visionapi_20240225_031226'

# 読み取り結果のリストを読み込む
with open(f'{base_dir}/recognize_list.txt', 'r') as file:
    lines = file.readlines()
    recognize_results = [line.strip().split(': ')[1] for line in lines]

# 画像ファイルのパスを生成
image_paths = [f'../../media/cropped/RouletteVideo_20240224/cropped_frame_{i}.jpg' for i in range(81)]

# 9x9のグリッドで可視化
fig, axs = plt.subplots(9, 9, figsize=(20, 20))

for i, ax in enumerate(axs.flat):
    # 画像を読み込み
    img = mpimg.imread(image_paths[i])
    ax.imshow(img)
    ax.axis('off')  # 軸を非表示にする
    # 認識結果をタイトルとして設定
    ax.set_title(recognize_results[i], fontsize=10)

plt.tight_layout()

# 可視化した画像を保存
plt.savefig(f'{base_dir}/visualization_result.png')