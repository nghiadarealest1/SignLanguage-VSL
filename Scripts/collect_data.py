import cv2
import os
import numpy as np
import mediapipe as mp
import time
import winsound  # chỉ dùng được trên Windows

# =====================================================
# 1. DANH SÁCH ACTION – NGÔN NGỮ KÝ HIỆU BỆNH NHÂN
# =====================================================
actions = np.array([
    # --- Giao tiếp cơ bản ---

    # --- Lấy số – chờ khám ---

    # --- Triệu chứng chung ---
    "uong",
    #
    # # --- Vị trí đau ---
    # "dau_dau", "dau_co", "dau_hong", "dau_nguc",
    # "dau_bung", "dau_lung", "dau_tay", "dau_chan",
    #
    # # --- Mức độ & cảm giác ---
    # "nhe", "nang", "nhieu", "it",
    # "dau_nhieu", "kho_chiu",
    #
    # # --- Thời gian ---
    # "hom_nay", "hom_qua", "may_ngay_nay",
    # "lau_roi", "vua_moi",
    #
    # # --- Thuốc & điều trị (góc nhìn bệnh nhân) ---


    #
    # # --- Kết thúc ---
    # "toi_hieu_roi", "toi_muon_hoi",
    # "toi_muon_ve", "cam_on_bac_si"
])

num_sequences = 30          # mỗi action quay 30 sequence
sequence_length = 100       # mỗi sequence 100 frame
DATA_PATH = os.path.join("../data")

# =====================================================
# 2. KHỞI TẠO MEDIAPIPE
# =====================================================
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# =====================================================
# 3. TRÍCH XUẤT KEYPOINTS
# Pose (99) + 2 tay (126) = 225
# =====================================================
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z]
                     for res in results.pose_landmarks.landmark]).flatten() \
           if results.pose_landmarks else np.zeros(99)

    lh = np.array([[res.x, res.y, res.z]
                   for res in results.left_hand_landmarks.landmark]).flatten() \
         if results.left_hand_landmarks else np.zeros(63)

    rh = np.array([[res.x, res.y, res.z]
                   for res in results.right_hand_landmarks.landmark]).flatten() \
         if results.right_hand_landmarks else np.zeros(63)

    return np.concatenate([pose, lh, rh])

# =====================================================
# 4. TẠO THƯ MỤC DATASET
# =====================================================
for action in actions:
    for seq in range(num_sequences):
        os.makedirs(os.path.join(DATA_PATH, action, str(seq)), exist_ok=True)

# =====================================================
# 5. THU THẬP DỮ LIỆU
# =====================================================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Không thể mở webcam")
    exit()

print("✅ Bắt đầu thu dữ liệu ngôn ngữ ký hiệu BỆNH NHÂN")

log_file = open("../collect_log.txt", "a", encoding="utf-8")
log_file.write(f"\n=== PHIÊN THU: {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")

with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as holistic:

    for action in actions:
        print(f"\n🔹 ACTION: {action.upper()}")

        for seq in range(num_sequences):

            # ---- Đếm ngược ----
            for countdown in range(3, 0, -1):
                ret, frame = cap.read()
                if not ret:
                    continue
                cv2.putText(frame,
                            f"Chuan bi: {action} ({countdown})",
                            (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.2, (0, 255, 255), 3)
                cv2.imshow("Collect Data", frame)
                cv2.waitKey(1000)

            winsound.Beep(1000, 300)
            print(f"▶ Thu {action} | Seq {seq+1}/{num_sequences}")

            valid_frames = 0

            for _ in range(sequence_length):
                ret, frame = cap.read()
                if not ret:
                    continue

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = holistic.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # ---- LỌC FRAME RÁC ----
                if results.pose_landmarks is None:
                    continue

                if results.left_hand_landmarks is None and results.right_hand_landmarks is None:
                    continue

                # ---- VẼ KEYPOINT ----
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks,
                    mp_holistic.POSE_CONNECTIONS
                )
                mp_drawing.draw_landmarks(
                    image, results.left_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS
                )
                mp_drawing.draw_landmarks(
                    image, results.right_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS
                )

                cv2.putText(
                    image,
                    f"{action} | Seq {seq+1} | Frame {valid_frames+1}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2
                )

                cv2.imshow("Collect Data", image)

                keypoints = extract_keypoints(results)
                np.save(
                    os.path.join(DATA_PATH, action, str(seq), str(valid_frames)),
                    keypoints
                )

                valid_frames += 1

                if cv2.waitKey(10) & 0xFF == ord("q"):
                    log_file.write("⛔ Thoát thủ công\n")
                    cap.release()
                    cv2.destroyAllWindows()
                    log_file.close()
                    exit()

            winsound.Beep(600, 200)
            log_file.write(
                f"{action} | Seq {seq} | Frames hợp lệ: {valid_frames}\n"
            )
            print(f"✅ Hoàn thành | Frames hợp lệ: {valid_frames}")

print("🎉 HOÀN TẤT THU DỮ LIỆU")
log_file.write("=== KẾT THÚC ===\n")
log_file.close()
cap.release()
cv2.destroyAllWindows()
