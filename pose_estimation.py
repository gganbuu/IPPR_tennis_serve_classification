import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Load the model
model = hub.load('https://tfhub.dev/google/movenet/singlepose/lightning/4')
movenet = model.signatures['serving_default']

@tf.function(experimental_relax_shapes=True)
def run_inference(input_image):
    outputs = movenet(input_image)
    return outputs['output_0']

def process_image(image_path, output_path):
    # Read the image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize and pad the image
    input_image = tf.image.resize_with_pad(image, 192, 192)
    input_image = tf.cast(input_image, dtype=tf.int32)
    input_image = tf.expand_dims(input_image, axis=0)

    # Run inference
    outputs = run_inference(input_image)
    keypoints = outputs.numpy()[0, 0, :, :2]

    # Define the indices for the keypoints we want to visualize
    desired_keypoints_indices = [5, 6, 7, 11, 12, 13, 14]  # Shoulders, Hands, Elbows, Hips, Knees, Feet

    # Draw only the desired keypoints
    height, width, _ = image.shape
    filtered_keypoints = []
    
    for idx in desired_keypoints_indices:
        kp = keypoints[idx]
        x, y = int(kp[1] * width), int(kp[0] * height)
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)  # Green circle
        cv2.circle(image, (x, y), 7, (0, 0, 0), 2)  # Black outline
        filtered_keypoints.append(kp)

    # Save the output image
    cv2.imwrite(str(output_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    return np.array(filtered_keypoints)

def process_images_in_folder(folder_path, output_folder):
    # Create output folder if it doesn't exist
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # Get list of all jpg files in the folder
    image_files = list(Path(folder_path).glob('*.jpg'))

    # Create a progress bar
    pbar = tqdm(image_files, desc=f"Processing {Path(folder_path).name}")

    # Iterate through all jpg files in the folder
    for img_path in pbar:
        try:
            # Get the keypoints
            keypoints = process_image(img_path, output_folder / f"{img_path.stem}_pose.jpg")

            # Save keypoint data for filtered keypoints
            keypoint_file = output_folder / f"{img_path.stem}_keypoints.txt"
            np.savetxt(str(keypoint_file), keypoints)

            # Update progress bar description
            pbar.set_postfix({'file': img_path.name, 'status': 'success'})
        except Exception as e:
            print(f"\nError processing {img_path}: {str(e)}")
            pbar.set_postfix({'file': img_path.name, 'status': 'error'})

def main():
    # Define the paths to your dataset folders
    dataset_root = Path('datasets') / 'tennis serve.v2i.coco(1)'/'Baddataset'
    # Define the folders for your dataset
    folders = ['WA00', 'WA01', 'WA02','WA03','WA04']

    print(f"Dataset root: {dataset_root.absolute()}")
    if not dataset_root.exists():
        print(f"Error: Dataset root folder does not exist: {dataset_root.absolute()}")
        return

    # Process images in each folder
    for folder in folders:
        input_folder = dataset_root / folder
        output_folder = dataset_root / f"{folder}_pose_output"
        
        if not input_folder.exists():
            print(f"\nError: Folder does not exist: {input_folder.absolute()}")
            continue
        
        jpg_count = len(list(input_folder.glob('*.jpg')))
        print(f"\nProcessing images in {folder} folder...")
        print(f"Found {jpg_count} JPG files in {input_folder.absolute()}")
        
        if jpg_count == 0:
            print(f"No JPG files found in {input_folder.absolute()}. Skipping.")
            continue
        
        process_images_in_folder(input_folder, output_folder)

    print("\nProcessing complete!")

if __name__ == "__main__":
    main()
