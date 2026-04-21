import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# ===================== CẤU HÌNH =====================
DATA_PATH = "../data"
SEQUENCE_LENGTH = 100

# CHỈ POSE + 2 HAND = 225 KEYPOINTS
KEYPOINTS_DIM = 225


# ===================== 1. LOAD DATASET =====================
def load_dataset():
    expected_keypoint_size = KEYPOINTS_DIM

    X = []
    y = []
    label_map = {}

    actions = sorted([f for f in os.listdir(DATA_PATH)
                      if os.path.isdir(os.path.join(DATA_PATH, f))])
    print(f"Tìm thấy {len(actions)} hành động: {actions}")

    for label_idx, action in enumerate(actions):
        label_map[label_idx] = action
        action_path = os.path.join(DATA_PATH, action)

        sequences = [d for d in os.listdir(action_path)
                     if os.path.isdir(os.path.join(action_path, d))]

        print(f"\n→ Đang xử lý: {action} ({len(sequences)} sequences)")
        loaded_count = 0

        for seq in sequences:
            seq_path = os.path.join(action_path, seq)
            npy_files = sorted([f for f in os.listdir(seq_path) if f.endswith('.npy')],
                               key=lambda x: int(os.path.splitext(x)[0]))

            if not npy_files:
                print(f"  [SKIP] Thư mục rỗng: {seq}")
                continue

            sequence = []
            for npy_file in npy_files:
                file_path = os.path.join(seq_path, npy_file)
                try:
                    kp = np.load(file_path)

                    if kp.shape[0] != expected_keypoint_size:
                        old = kp.shape[0]
                        if kp.shape[0] < expected_keypoint_size:
                            pad = np.zeros(expected_keypoint_size - kp.shape[0])
                            kp = np.concatenate([kp, pad])
                        else:
                            kp = kp[:expected_keypoint_size]
                        print(f"[FIX] {file_path}: {old} → {expected_keypoint_size}")

                    sequence.append(kp)

                except Exception as e:
                    print(f"[ERROR] Load lỗi {file_path}: {e}")

            if len(sequence) == 0:
                continue

            # Chuẩn hóa thành đúng 100 frame
            if len(sequence) > SEQUENCE_LENGTH:
                sequence = sequence[:SEQUENCE_LENGTH]
            else:
                last = sequence[-1]
                while len(sequence) < SEQUENCE_LENGTH:
                    sequence.append(last.copy())

            X.append(sequence)
            y.append(label_idx)
            loaded_count += 1

        print(f"{action}: load thành công {loaded_count}/{len(sequences)} sequences")

    X = np.array(X)
    y = np.array(y)

    print("\nLOAD HOÀN TẤT!")
    print(f"Tổng sample: {len(X)}")
    print(f"Shape X: {X.shape}")
    print(f"Số lớp: {len(label_map)}")

    return X, y, label_map


# ===================== 2. BUILD MODEL =====================
def build_model(num_classes):
    model = Sequential([
        Input(shape=(SEQUENCE_LENGTH, KEYPOINTS_DIM)),
        LSTM(128, return_sequences=True),
        Dropout(0.3),

        LSTM(128, return_sequences=False),
        Dropout(0.3),

        Dense(128, activation='relu'),
        Dropout(0.3),

        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()
    return model


# ===================== 3. MAIN =====================
def main():
    # Load dữ liệu
    X, y_raw, label_map = load_dataset()

    num_classes = len(label_map)
    y = to_categorical(y_raw, num_classes=num_classes)

    print(f"One-hot y → {y.shape}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y_raw
    )
    print(f"Train: {X_train.shape[0]} mẫu | Test: {X_test.shape[0]} mẫu")

    # Build & Train
    model = build_model(num_classes)
    print("\nBắt đầu huấn luyện...\n")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=16,
        verbose=1
    )

    # Save model + label map
    model.save("gesture_model.keras")
    np.save("../Models/label_map.npy", label_map)
    print("\nĐã lưu model và label map\n")


    # ========================= BIỂU ĐỒ & THỐNG KÊ =========================
    import matplotlib.pyplot as plt
    from sklearn.metrics import (
        confusion_matrix, ConfusionMatrixDisplay,
        classification_report, f1_score
    )

    # ----- Accuracy curve -----
    plt.figure(figsize=(6, 4))
    plt.plot(history.history["accuracy"], label="Train Acc")
    plt.plot(history.history["val_accuracy"], label="Val Acc")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig("accuracy_curve.png", dpi=300)
    plt.close()

    # ----- Loss curve -----
    plt.figure(figsize=(6, 4))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss_curve.png", dpi=300)
    plt.close()

    # ========================= CONFUSION MATRIX =========================
    print("\nTạo confusion matrix...")

    y_true = np.argmax(y_test, axis=1)
    y_pred = np.argmax(model.predict(X_test), axis=1)

    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[label_map[i] for i in range(num_classes)]
    )
    disp.plot(cmap='Blues', xticks_rotation=45)
    plt.title("Ma trận nhầm lẫn mô hình LSTM")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=300)
    plt.close()

    # ========================= CLASSIFICATION REPORT =========================
    report = classification_report(
        y_true, y_pred,
        target_names=[label_map[i] for i in range(num_classes)]
    )
    with open("../classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    # ========================= F1-SCORE BAR CHART =========================
    f1_scores = f1_score(y_true, y_pred, average=None)

    plt.figure(figsize=(7, 4))
    plt.bar([label_map[i] for i in range(num_classes)], f1_scores)
    plt.title("F1-score theo từng lớp")
    plt.ylabel("F1-score")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("f1_score_chart.png", dpi=300)
    plt.close()

    print("\n=== HOÀN TẤT! ===")
    print("Đã tạo các file:")
    print(" - accuracy_curve.png")
    print(" - loss_curve.png")
    print(" - confusion_matrix.png")
    print(" - f1_score_chart.png")
    print(" - classification_report.txt")


if __name__ == "__main__":
    main()
