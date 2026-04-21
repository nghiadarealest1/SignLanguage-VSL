import os
import numpy as np
from sklearn.model_selection import train_test_split

DATA_PATH = '../data'
sequence_length = 100

# Lấy danh sách action (sắp xếp để cố định thứ tự)
actions = np.array(sorted([
    folder for folder in os.listdir(DATA_PATH)
    if os.path.isdir(os.path.join(DATA_PATH, folder))
]))

print("Danh sách action:", actions)

X_all, y_all = [], []

for action_idx, action in enumerate(actions):
    print(f"\n=== Đang xử lý: {action} ===")
    action_path = os.path.join(DATA_PATH, action)

    sequence_folders = [
        f for f in os.listdir(action_path)
        if os.path.isdir(os.path.join(action_path, f))
    ]

    print(f"  Tìm thấy {len(sequence_folders)} sequence")
    valid_sequences = 0

    for seq in sequence_folders:
        seq_path = os.path.join(action_path, seq)
        npy_files = [f for f in os.listdir(seq_path) if f.endswith('.npy')]

        if len(npy_files) == 0:
            print(f"  Bỏ qua (rỗng): {seq_path}")
            continue

        # Sắp xếp an toàn theo số trong tên file
        try:
            npy_files = sorted(npy_files, key=lambda x: int(os.path.splitext(x)[0]))
        except:
            npy_files = sorted(npy_files)

        sequence = []
        for f in npy_files:
            keypoints = np.load(os.path.join(seq_path, f))
            sequence.append(keypoints)

        current_len = len(sequence)
        print(f"  {seq}: {current_len} frame")

        if current_len >= sequence_length:
            sequence = sequence[:sequence_length]
        else:
            while len(sequence) < sequence_length:
                sequence.append(sequence[-1])

        X_all.append(sequence)
        y_all.append(action_idx)
        valid_sequences += 1

    print(f"✔ {action}: giữ lại {valid_sequences} sequence hợp lệ")

# Chuyển sang numpy
X = np.array(X_all)
y = np.array(y_all)

print(f"\nTổng số sample: {len(X)}")
print(f"Shape X: {X.shape}")  # (n, 100, feature_dim)

# Kiểm tra shape
if X.ndim != 3 or X.shape[1] != 100:
    raise ValueError(f"Shape X sai: {X.shape}. Phải là (n, 100, feature_dim)")

# Chia train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Lưu dataset
SAVE_PATH = "../dataset"
os.makedirs(SAVE_PATH, exist_ok=True)

np.save(os.path.join(SAVE_PATH, "X_train.npy"), X_train)
np.save(os.path.join(SAVE_PATH, "X_test.npy"), X_test)
np.save(os.path.join(SAVE_PATH, "y_train.npy"), y_train)
np.save(os.path.join(SAVE_PATH, "y_test.npy"), y_test)
np.save(os.path.join(SAVE_PATH, "actions.npy"), actions)

print("\n✅ HOÀN TẤT TẠO DATASET")
print(f"X_train: {X_train.shape}")
print(f"X_test : {X_test.shape}")
print(f"Số lớp: {len(actions)} → {actions}")
