import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

# Load model và label map
model = tf.keras.models.load_model("../Models/gesture_model.keras")
label_map = np.load("../Models/label_map.npy", allow_pickle=True).item()

SEQUENCE_LENGTH = 100
KEYPOINTS_DIM = 225

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


# ===== Extract 225 keypoints: pose(99) + left(63) + right(63) =====
def extract_keypoints(results):
    pose = np.array(
        [[res.x, res.y, res.z]
         for res in results.pose_landmarks.landmark]
    ).flatten() if results.pose_landmarks else np.zeros(33*3)

    lh = np.array(
        [[res.x, res.y, res.z]
         for res in results.left_hand_landmarks.landmark]
    ).flatten() if results.left_hand_landmarks else np.zeros(21*3)

    rh = np.array(
        [[res.x, res.y, res.z]
         for res in results.right_hand_landmarks.landmark]
    ).flatten() if results.right_hand_landmarks else np.zeros(21*3)

    return np.concatenate([pose, lh, rh])  # = 225 giá trị


# ===== RUN REAL-TIME =====
sequence = []
cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as holistic:

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img.flags.writeable = False
        results = holistic.process(img)
        img.flags.writeable = True
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Vẽ landmarks
        mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(img, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(img, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # Extract 225 keypoints
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)

        # Giữ đúng 100 frames
        if len(sequence) > SEQUENCE_LENGTH:
            sequence.pop(0)

        # Khi đủ 100 frames → dự đoán
        if len(sequence) == SEQUENCE_LENGTH:
            X_input = np.expand_dims(sequence, axis=0)
            prediction = model.predict(X_input)[0]

            action = label_map[np.argmax(prediction)]
            prob = np.max(prediction)

            cv2.putText(img, f"{action} ({prob:.2f})",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0, 255, 0), 3)

        cv2.imshow("Real-Time Benh_an", img)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
