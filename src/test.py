from detect_numbers_visionapi import process_image

def process_frames(start_frame, end_frame, image_dir='../media/frame/RouletteVideo_20240224/', top_left=(1240, 560), bottom_right=(1310, 640)):
    for frame_number in range(start_frame, end_frame + 1):
        image_path = f"{image_dir}frame_{frame_number}.jpg"
        print(f"Processing {image_path}...")
        process_image(image_path, top_left, bottom_right)

if __name__ == "__main__":
    process_frames(0, 81)