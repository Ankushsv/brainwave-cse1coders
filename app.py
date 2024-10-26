from flask import Flask, render_template, Response, request,redirect,url_for
import cv2
import mediapipe as mp
import numpy as np

app = Flask(__name__)

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.7)

# Default t-shirt image
tshirt_img = None
# tshirt_path = 'static/t-shirt_template.png'
alpha = 0.5
previous_points = None

# Load the selected t-shirt image
def load_tshirt(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Error: T-shirt image not found at {image_path}")
    return img

# Smooth key points for stable overlay
def smooth_points(current_points, previous_points):
    smoothed = []
    for (cx, cy), (px, py) in zip(current_points, previous_points):
        sx = alpha * cx + (1 - alpha) * px
        sy = alpha * cy + (1 - alpha) * py
        smoothed.append([sx, sy])
    return smoothed

# Overlay the t-shirt onto the webcam frame
def apply_tshirt(frame, tshirt_img, pts_src, pts_dst):
    M = cv2.getPerspectiveTransform(np.float32(pts_src), np.float32(pts_dst))
    rows, cols, _ = frame.shape
    warped_tshirt = cv2.warpPerspective(tshirt_img, M, (cols, rows), borderMode=cv2.BORDER_TRANSPARENT)

    alpha_channel = warped_tshirt[:, :, 3] / 255.0
    for c in range(3):
        frame[:, :, c] = frame[:, :, c] * (1 - alpha_channel) + warped_tshirt[:, :, c] * alpha_channel

    return frame

# Generate webcam frames with t-shirt overlay
def generate_frames():
    global previous_points, tshirt_img

    # if tshirt_img is None:
    #     tshirt_img = load_tshirt(tshirt_path)  # Load default t-shirt

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            current_points = [
                [int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * frame.shape[1]),
                 int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * frame.shape[0])],
                [int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * frame.shape[1]),
                 int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * frame.shape[0])],
                [int(landmarks[mp_pose.PoseLandmark.LEFT_HIP].x * frame.shape[1]),
                 int(landmarks[mp_pose.PoseLandmark.LEFT_HIP].y * frame.shape[0])],
                [int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x * frame.shape[1]),
                 int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y * frame.shape[0])]
            ]

            # Calculate shoulder width and resize the t-shirt accordingly
            left_shoulder, right_shoulder = current_points[0], current_points[1]
            shoulder_width = int(np.linalg.norm(np.array(left_shoulder) - np.array(right_shoulder)))

            desired_tshirt_width = shoulder_width
            aspect_ratio = tshirt_img.shape[0] / tshirt_img.shape[1]
            desired_tshirt_height = int(desired_tshirt_width * aspect_ratio)

            resized_tshirt = cv2.resize(tshirt_img, (desired_tshirt_width, desired_tshirt_height), interpolation=cv2.INTER_AREA)

            if previous_points is not None:
                smoothed_points = smooth_points(current_points, previous_points)
            else:
                smoothed_points = current_points

            previous_points = smoothed_points

            pts_src = np.float32([[0, 0], [resized_tshirt.shape[1], 0],
                                  [0, resized_tshirt.shape[0]], [resized_tshirt.shape[1], resized_tshirt.shape[0]]])
            pts_dst = np.float32(smoothed_points)

            frame = apply_tshirt(frame, resized_tshirt, pts_src, pts_dst)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('home.html')

@app.route("/collection")
def collection():
    return render_template("collection.html")


@app.route('/change_tshirt', methods=['POST','GET'])
def change_tshirt():
    global tshirt_img
    tshirt_choice = request.form.get('tshirt')
    tshirt_path = f'static/{tshirt_choice}.png'
    tshirt_img = load_tshirt(tshirt_path)
    return redirect(url_for('video_feed'))

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
