# Nghiên cứu một số công cụ trong thị giác máy tính và mô hình học sâu để xây dựng hệ thống phục vụ giao tiếp giữa nhân viên y tế và bệnh nhân khiếm thính tại các bệnh viện
## 📌 Giới thiệu
Ngôn ngữ ký hiệu được xem là “cầu nối” giao tiếp chính của người khiếm thính. Tuy nhiên, do đa số nhân viên y tế không được đào tạo sử dụng ngôn ngữ ký hiệu, nên việc hiểu và phản hồi chính xác nội dung của bệnh nhân là điều không hề dễ dàng. Chính vì vậy, việc xây dựng một hệ thống thông minh có khả năng nhận diện và chuyển đổi ngôn ngữ ký hiệu thành tiếng Việt sẽ góp phần thu hẹp khoảng cách giao tiếp, mang lại sự công bằng trong tiếp cận dịch vụ y tế cho nhóm người yếu thế này.
	Từ những cơ sở thực tiễn và khoa học nêu trên, đề tài “Nghiên cứu một số công cụ trong thị giác máy tính và mô hình học sâu để xây dựng hệ thống phục vụ giao tiếp giữa nhân viên y tế và bệnh nhân khiếm thính tại các bệnh viện” được lựa chọn với mong muốn đóng góp một giải pháp công nghệ khả thi và mang tính nhân văn, đồng thời thể hiện khả năng vận dụng kiến thức chuyên ngành vào giải quyết các vấn đề xã hội thực tế.
## Thu thập dữ liệu
Trong nghiên cứu này, việc thu thập dữ liệu được thực hiện dựa trên mô hình Holistic của thư viện MediaPipe, có khả năng trích xuất đồng thời các đặc trưng tư thế và bàn tay từ hình ảnh hoặc video.
	Mỗi video ngôn ngữ ký hiệu (ví dụ “bác sĩ”, “bệnh nhân”, “dị ứng”) được xử lý thành chuỗi khung hình (frame), và từ mỗi khung hình, chương trình sẽ trích xuất các điểm mốc (landmarks) gồm:
- 33 điểm cơ thể (pose landmarks)
- 21 điểm tay trái và 21 điểm tay phải (hand landmarks)
	Tổng cộng, mỗi frame chứa hơn 540 điểm đặc trưng, được lưu trữ dưới dạng mảng số học ba chiều (x, y, z) phản ánh vị trí của từng bộ phận trong không gian.
	Dữ liệu đầu ra của mỗi frame được lưu dưới định dạng .npy (NumPy array file) — một định dạng phổ biến trong khoa học dữ liệu, giúp lưu trữ mảng nhiều chiều với tốc độ truy xuất cao, đồng thời dễ dàng nạp lại trong quá trình huấn luyện mô hình.
Sau khi hoàn tất quá trình thu thập, dữ liệu được sắp xếp trong thư mục data/ theo cấu trúc:
<img width="413" height="520" alt="image" src="https://github.com/user-attachments/assets/352b5142-8ece-41ac-97aa-21e463b7cd94" />
Quá trình thu thập dữ liệu được triển khai thông qua một chương trình Python có chức năng ghi nhận chuỗi khung hình (frames) tương ứng với từng hành động ký hiệu. Mỗi phiên thu thập được tiến hành như sau:
- Người thực hiện đứng hoặc ngồi trước camera trong điều kiện ánh sáng ổn định.
- Mỗi hành động (ví dụ: uống, đau bụng, xin chào, ...) được thực hiện trong nhiều lần lặp lại để tăng tính đa dạng.
- Với mỗi hành động, hệ thống ghi nhận 100 khung hình liên tiếp, đây là độ dài chuẩn của một chuỗi cử chỉ để mô hình LSTM có thể học được sự thay đổi theo thời gian.
- Số lần lặp (sequences) được đặt mặc định là 30, nhằm đảm bảo mỗi hành động có đủ mẫu đại diện cho nhiều trạng thái tay, vị trí, tốc độ và góc quay khác nhau.
	Trong quá trình ghi dữ liệu, hệ thống hiển thị trực quan lên màn hình gồm:
- Khung hình camera theo thời gian thực.
- Keypoints cơ thể, mặt và bàn tay được MediaPipe nhận diện.
- Thanh tiến trình sequence và frame số bao nhiêu.
- Âm báo để hỗ trợ người thực hiện bắt đầu tạo dữ liệu đúng thời điểm.
## Tiền xử lý dữ liệu
Quá trình chuẩn hóa bắt đầu ngay khi thu thập dữ liệu từ camera trong file collect_data.py. Tại mỗi khung hình, MediaPipe Holistic tạo ra ba nhóm điểm mốc chính:
- Pose Landmarks: 33 điểm, mỗi điểm gồm 4 giá trị (x, y, z, visibility)
- Hand Landmarks (trái và phải): mỗi bên 21 điểm, mỗi điểm gồm 3 giá trị (x, y, z)
