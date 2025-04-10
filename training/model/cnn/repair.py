import os
from PIL import Image
from config import get_cnn_image_dir
from tqdm import tqdm

# Match this with your dataset config
IMAGE_OUTPUT_SIZE = (1300, 1600)


def repair_images(image_dir: str, expected_size=(1300, 1600)):
    all_files = [f for f in os.listdir(image_dir) if f.endswith(".png")]
    print(f"Found {len(all_files)} images to check in: {image_dir}")

    for file in tqdm(all_files):
        path = os.path.join(image_dir, file)
        try:
            img = Image.open(path).convert("L")

            if img.size != expected_size[::-1]:  # PIL has (W, H)
                print(f"Repairing {file}: {img.size} → {expected_size}")
                img = img.resize(expected_size)
                img.save(path)

        except Exception as e:
            print(f"⚠️ Failed to process {file}: {e}")


if __name__ == "__main__":
    image_dir = get_cnn_image_dir(IMAGE_OUTPUT_SIZE)
    repair_images(image_dir, expected_size=IMAGE_OUTPUT_SIZE)
