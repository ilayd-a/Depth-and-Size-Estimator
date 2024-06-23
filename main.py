# created by ilayd 06/22/24
''' @ARTICLE {Ranftl2022,
    author  = "Ren\'{e} Ranftl and Katrin Lasinger and David Hafner and Konrad Schindler and Vladlen Koltun",
    title   = "Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-Shot Cross-Dataset Transfer",
    journal = "IEEE Transactions on Pattern Analysis and Machine Intelligence",
    year    = "2022",
    volume  = "44",
    number  = "3"
}

@article{Ranftl2021,
	author    = {Ren\'{e} Ranftl and Alexey Bochkovskiy and Vladlen Koltun},
	title     = {Vision Transformers for Dense Prediction},
	journal   = {ICCV},
	year      = {2021},
}
'''
import torch
import cv2
import numpy as np
import ssl

"""
   Draws bounding boxes on the frame with object labels and depth information.

   Parameters:
   - frame: The video frame to draw on.
   - detections: Detected objects with their bounding box coordinates, confidence scores, and class IDs.
   - depth_map: Depth information corresponding to the frame.

   Returns:
   - frame: The frame with bounding boxes and labels drawn.
   """
def draw_bounding_boxes(frame, detections, depth_map):
    for *box, conf, cls in detections:
        x1, y1, x2, y2 = map(int, box)
        label = yolov5.names[int(cls)] # Get the class label from YOLOv5 model
        width = x2 - x1 # Width of the bounding box
        height = y2 - y1 # Height of the bounding box
        depth = np.mean(depth_map[y1:y2, x1:x2]) # Average depth within the bounding box
        # Draw the bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        # Put the label and depth information above the bounding box
        cv2.putText(frame, f'{label} {depth:.2f}m', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (36, 255, 12), 2)
        # Put the size of the bounding box below the top-left corner
        cv2.putText(frame, f'Size: {width}x{height}', (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (36, 255, 12), 2)
    return frame

# disable ssl certificate verification (not recommended but it wasn't working without this for me)
ssl._create_default_https_context = ssl._create_unverified_context

# Load MiDaS model for depth estimation
midas = torch.hub.load('intel-isl/MiDaS', 'DPT_Large')
device = "cpu" # Use CPU for computation
midas.to(device)
midas.eval() # Set the model to evaluation mode

# Load MiDaS transforms for preprocessing the input frame
midasTransforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
transform = midasTransforms.dpt_transform

# Load YOLOv5 model for object detection
yolov5 = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=True).to(device)

# Open a connection to the webcam (0 is the default camera)
video = cv2.VideoCapture(0)


while video.isOpened():
    ret, frame = video.read()  # Read a frame from the video capture
    if not ret or frame is None:
        print("Error: Could not read frame or it is empty.")
        break

    # Convert the frame from BGR to RGB (required by many models)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Transform the frame for MiDaS model input
    input_batch = transform(frame_rgb).to(device)

    with torch.no_grad():   # Disable gradient computation for efficiency
        prediction = midas(input_batch) # Get the depth prediction from MiDaS
        # Resize the depth map to match the frame size
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame.shape[:2],
            mode='bicubic',
            align_corners=False,
        ).squeeze()

        depth_map = prediction.cpu().numpy() # Convert the depth map to a NumPy array

    # Perform object detection using YOLOv5
    results = yolov5(frame_rgb)
    detections = results.xyxy[0].cpu().numpy() # Extract detections and convert to NumPy array

    # Draw bounding boxes and depth information on the frame
    frame_with_boxes = draw_bounding_boxes(frame, detections, depth_map)

    # Display the frame with bounding boxes in a window
    cv2.imshow('Real-time Object Detection and Depth Estimation', frame_with_boxes)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
video.release()
cv2.destroyAllWindows()