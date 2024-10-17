import cv2
import os

# Path to your video file
video_path = '/Users/rajkulkarni/Desktop/Vishu/BS_4.mp4'
output_folder = 'extracted_frames/BS_0/'

# Create the output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Open the video file
video_capture = cv2.VideoCapture(video_path)

# Check if the video was opened successfully
if not video_capture.isOpened():
    print(f"Error: Could not open video {video_path}")
    exit()

# Get the total number of frames and the frame rate (fps)
fps = video_capture.get(cv2.CAP_PROP_FPS)
total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"FPS: {fps}, Total frames: {total_frames}")

# Initialize frame counter
frame_number = 0

# Loop through the video and save each frame
while True:
    success, frame = video_capture.read()
    
    # If the video has ended, break the loop
    if not success:
        break

    # Save frame as an image file
    frame_filename = os.path.join(output_folder, f"BS_0_v4_frame_{frame_number:04d}.jpg")
    cv2.imwrite(frame_filename, frame)
    
    frame_number += 1

# Release the video capture object
video_capture.release()
print(f"Frames have been extracted to: {output_folder}")
