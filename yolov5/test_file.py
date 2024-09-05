import cv2
import torch

# Load YOLOv5 model (make sure to provide the correct path to your best.pt file)
model = torch.hub.load('ultralytics/yolov5', 'custom', path=r'yolov5/runs/train/exp/weights/best.pt')

# Input and output paths
input_video_path = r'yolov5\videos\Eyewitness Videos of Paris Shooting Terror Attack at Charlie Hebdo _ The New York Times.mp4'
output_video_path = r'yolov5\videos\output.mp4'

# Open the input video
cap = cv2.VideoCapture(input_video_path)

# Get the video frame width, height, and FPS
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Process the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference on the frame
    results = model(frame)

    # Render results on the frame
    frame = results.render()[0]

    # Write the frame to the output video
    out.write(frame)

# Release video objects
cap.release()
out.release()

print(f"Processed video saved to {output_video_path}")
