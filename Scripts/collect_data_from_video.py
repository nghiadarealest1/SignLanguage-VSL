import cv2
import numpy as np
import mediapipe as mp
import os

# =============================
# 1. Cấu hình
# =============================
action = 'buon_non'              # tên hành động (ký hiệu)
DATA_PATH = '../data'
sequence_length = 100          # số frame cần lấy

# Thư mục lưu dữ liệu (data/action/from_video)
save_path = os.path.join(DATA_PATH, action, 'from_video')
os.makedirs(save_path, exist_ok=True)

# Đường dẫn đến video đầu vào
video_path = os.path.join('../Videos', 'buon_non.mp4')
if not os.path.exists(video_path):
    raise FileNotFoundError(f"Không tìm thấy video tại: {video_path}")

# =============================
# 2. Khởi tạo MediaPipe Holistic
# =============================
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# =============================
# 3. Hàm trích xuất keypoints
#    (POSE + 2 TAY, KHÔNG FACE)
# =============================
def extract_keypoints(results):
    # Pose: 33 × 3 = 99
    pose = np.array([[res.x, res.y, res.z]
                     for res in results.pose_landmarks.landmark]).flatten() \
           if results.pose_landmarks else np.zeros(99)

    # Left hand: 21 × 3 = 63
    lh = np.array([[res.x, res.y, res.z]
                   for res in results.left_hand_landmarks.landmark]).flatten() \
         if results.left_hand_landmarks else np.zeros(63)

    # Right hand: 21 × 3 = 63
    rh = np.array([[res.x, res.y, res.z]
                   for res in results.right_hand_landmarks.landmark]).flatten() \
         if results.right_hand_landmarks else np.zeros(63)

    # Tổng: 225 chiều
    return np.concatenate([pose, lh, rh])

# =============================
# 4. Đọc video & trích xuất
# =============================
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise Exception(f"❌ Không thể mở video: {video_path}")

frame_num = 0
with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
) as holistic:

    while cap.isOpened() and frame_num < sequence_length:
        ret, frame = cap.read()
        if not ret:
            break

        # MediaPipe xử lý
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Vẽ POSE + HANDS (KHÔNG FACE)
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # Trích xuất & lưu keypoints
        keypoints = extract_keypoints(results)
        np.save(os.path.join(save_path, str(frame_num)), keypoints)
        frame_num += 1

        # Hiển thị
        cv2.putText(
            image,
            f'{action.upper()} - Frame {frame_num}/{sequence_length}',
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )
        cv2.imshow('Video Collect', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print(f"✅ Đã thu {frame_num} frame từ video '{video_path}'")
print(f"📁 Lưu tại: {save_path}")
