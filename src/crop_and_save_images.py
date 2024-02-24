import cv2
import glob
import os

def crop_image(image_path, top_left, bottom_right):
    """
    画像を読み込み、指定された座標でクロップする関数
    """
    image = cv2.imread(image_path)
    cropped_image = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    return cropped_image

def main():
    # すべての画像ファイルのパスを取得
    image_files = glob.glob('../media/frame/RouletteVideo_20240224/*.jpg')

    # 新しいディレクトリのパスを定義
    new_dir_path = '../media/cropped/RouletteVideo_20240224/'
    os.makedirs(new_dir_path, exist_ok=True)  # ディレクトリがなければ作成

    for image_path in image_files:
        # 画像ファイル名を取得
        image_name = os.path.basename(image_path)
        # クロップした画像を保存するパス。ファイル名の前に 'cropped_' を追加
        cropped_image_path = os.path.join(new_dir_path, f"cropped_{image_name}")
        # 画像をクロップ
        cropped_image = crop_image(image_path, (1240, 560), (1310, 640))
        # クロップした画像を保存
        cv2.imwrite(cropped_image_path, cropped_image)
        print(f"Saved: {cropped_image_path}")

if __name__ == "__main__":
    main()