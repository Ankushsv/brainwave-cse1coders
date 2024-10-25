import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2

# Load pre-trained pose estimation model from TensorFlow Hub (e.g., MoveNet)
pose_model = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")

# Load T-shirt image (overlay) and preprocess
def load_tshirt_image(image_path):
  try:
    tshirt = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # Load with alpha channel
    if tshirt is None:
      raise FileNotFoundError(f"Failed to load image: {image_path}")
    tshirt = cv2.resize(tshirt, (150, 150))  # Resize as per requirement
    return tshirt
  except FileNotFoundError as e:
    print(f"Error: {e}")
    return None

tshirt_image = load_tshirt_image("tshirt_template.png")  # Transparent T-shirt image

# Detect pose and get key points
def detect_pose(image):
  original_height, original_width, _ = image.shape
  input_image = cv2.resize(image, (256, 256))
  input_image = tf.image.resize_with_pad(tf.expand_dims(input_image, axis=0), 256, 256)
  input_image = tf.cast(input_image, dtype=tf.int32)
  # Run model inference
  try:
    outputs = pose_model(input_image)
    keypoints = outputs["output_0"].numpy()[0, 0, :, :2]  # [num_keypoints, 2] - x, y coordinates
    return keypoints * np.array([original_width, original_height])  # Scale to original dimensions
  except Exception as e:
    print(f"Error during pose detection: {e}")
    return None

# Overlay T-shirt on torso
def overlay_tshirt(frame, tshirt_img, keypoints):
  if tshirt_img is None or keypoints is None:
    return
  left_shoulder = keypoints[5]  # Left shoulder
  right_shoulder = keypoints[6]  # Right shoulder
  mid_shoulder = ((left_shoulder[0] + right_shoulder[0]) // 2,
                  (left_shoulder[1] + right_shoulder[1]) // 2)

  # Compute T-shirt position and size based on shoulders
  width = int(np.linalg.norm(right_shoulder - left_shoulder) * 1.5)
  height = width  # Assume square T-shirt for simplicity
  overlay_img = cv2.resize(tshirt_img, (width, height))

  # Determine top-left position for overlay
  x_offset = int(mid_shoulder[0] - width // 2)
  y_offset = int(mid_shoulder[1])

  # Overlay the T-shirt with transparency
  for c in range(3):  # For each color channel
    frame[y_offset:y_offset + height, x_offset:x_offset + width, c] = (
        overlay_img[:, :, c] * (overlay_img[:, :, 3] / 255.0) +
        frame[y_offset:y_offset + height, x_offset:x_offset + width, c] * (1 - overlay_img[:, :, 3] / 255.0)
    )


# Capture video from webcam
cap = cv2.VideoCapture(0)
while cap.isOpened():
  ret, frame = cap.read()
  if not ret:
    break

  # Resize frame for faster processing