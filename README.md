🧠 Sign Language VSL - AI for Healthcare Communication
📌 Giới thiệu

Đây là đồ án tốt nghiệp ngành Công nghệ Thông tin tại Trường Đại học Kiến trúc Hà Nội.
Dự án tập trung nghiên cứu và xây dựng hệ thống ứng dụng Trí tuệ nhân tạo (AI) và Thị giác máy tính (Computer Vision) nhằm hỗ trợ giao tiếp giữa:

👨‍⚕️ Nhân viên y tế
🧏‍♂️ Bệnh nhân khiếm thính

Trong môi trường bệnh viện.

Mục tiêu chính: Giảm rào cản giao tiếp và nâng cao chất lượng chăm sóc y tế thông qua công nghệ.

🎯 Mục tiêu dự án
Nhận diện ngôn ngữ ký hiệu tiếng Việt (VSL) từ hình ảnh/video
Xây dựng mô hình AI sử dụng Deep Learning
Tích hợp hệ thống vào ứng dụng giao tiếp thời gian thực
Hỗ trợ chuyển đổi:
✋ Ký hiệu → 📝 Văn bản
📝 Văn bản → ✋ Gợi ý ký hiệu
🧩 Chức năng chính
🎥 Nhận diện ký hiệu tay qua webcam
💬 Chat giữa bác sĩ và bệnh nhân
🧠 Dự đoán hành động bằng mô hình AI
📊 Hiển thị kết quả trực quan
🔐 Đăng nhập / phân quyền người dùng
📡 Kết nối video (real-time communication)
🏗️ Công nghệ sử dụng
🔹 Ngôn ngữ & Framework
Python
TensorFlow / Keras
OpenCV
MediaPipe
🔹 Mô hình AI
CNN (Convolutional Neural Network)
LSTM (Long Short-Term Memory)
🔹 Công cụ khác
PyCharm
SQLite
Radmin VPN (kết nối mạng nội bộ)
🧠 Kiến trúc hệ thống

Hệ thống gồm 3 phần chính:

Thu thập dữ liệu
Video / hình ảnh ký hiệu
Trích xuất đặc trưng bằng MediaPipe
Xử lý & huấn luyện
Tiền xử lý dữ liệu
Huấn luyện mô hình CNN + LSTM
Ứng dụng thực tế
Nhận diện real-time
Giao diện người dùng (GUI)
Chat & video call
📊 Dataset

⚠️ Dataset không được đưa lên GitHub do giới hạn dung lượng.

👉 Cách sử dụng:

Tải dataset từ link (Google Drive hoặc nguồn khác)
Đặt vào thư mục:
dataset/
🚀 Cài đặt & chạy project
1. Clone repo
git clone https://github.com/nghiadarealest1/SignLanguage-VSL.git
cd SignLanguage-VSL
2. Cài thư viện
pip install -r requirements.txt
3. Chạy ứng dụng
python main.py
📈 Đánh giá

Mô hình được đánh giá dựa trên:

Accuracy
Loss
Confusion Matrix
Precision / Recall / F1-score

Hệ thống đạt hiệu quả tốt trong:

Nhận diện ký hiệu cơ bản
Hoạt động thời gian thực
⚠️ Hạn chế
Dataset còn hạn chế
Chưa hỗ trợ đầy đủ ngôn ngữ ký hiệu VSL
Chưa xử lý tốt chuỗi dài/phức tạp
🔮 Hướng phát triển
Mở rộng dataset
Tích hợp NLP (xử lý ngôn ngữ tự nhiên)
Nhận diện cảm xúc / khuôn mặt
Deploy thành ứng dụng thực tế (Web/App)
