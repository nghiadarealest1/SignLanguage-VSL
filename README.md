# Nghiên cứu một số công cụ trong thị giác máy tính và mô hình học sâu để xây dựng hệ thống phục vụ giao tiếp giữa nhân viên y tế và bệnh nhân khiếm thính tại các bệnh viện
## 📌 Giới thiệu
Ngôn ngữ ký hiệu được xem là “cầu nối” giao tiếp chính của người khiếm thính. Tuy nhiên, do đa số nhân viên y tế không được đào tạo sử dụng ngôn ngữ ký hiệu, nên việc hiểu và phản hồi chính xác nội dung của bệnh nhân là điều không hề dễ dàng. Chính vì vậy, việc xây dựng một hệ thống thông minh có khả năng nhận diện và chuyển đổi ngôn ngữ ký hiệu thành tiếng Việt sẽ góp phần thu hẹp khoảng cách giao tiếp, mang lại sự công bằng trong tiếp cận dịch vụ y tế cho nhóm người yếu thế này.
	Từ những cơ sở thực tiễn và khoa học nêu trên, đề tài “Nghiên cứu một số công cụ trong thị giác máy tính và mô hình học sâu để xây dựng hệ thống phục vụ giao tiếp giữa nhân viên y tế và bệnh nhân khiếm thính tại các bệnh viện” được lựa chọn với mong muốn đóng góp một giải pháp công nghệ khả thi và mang tính nhân văn, đồng thời thể hiện khả năng vận dụng kiến thức chuyên ngành vào giải quyết các vấn đề xã hội thực tế.
## Thu thập dữ liệu
Trong nghiên cứu này, việc thu thập dữ liệu được thực hiện dựa trên mô hình Holistic của thư viện MediaPipe, có khả năng trích xuất đồng thời các đặc trưng tư thế và bàn tay từ hình ảnh hoặc video.
	Mỗi video ngôn ngữ ký hiệu (ví dụ “bác sĩ”, “bệnh nhân”, “dị ứng”) được xử lý thành chuỗi khung hình (frame), và từ mỗi khung hình, chương trình sẽ trích xuất các điểm mốc (landmarks) gồm:
33 điểm cơ thể (pose landmarks)
21 điểm tay trái và 21 điểm tay phải (hand landmarks)
	Tổng cộng, mỗi frame chứa hơn 540 điểm đặc trưng, được lưu trữ dưới dạng mảng số học ba chiều (x, y, z) phản ánh vị trí của từng bộ phận trong không gian.
	Dữ liệu đầu ra của mỗi frame được lưu dưới định dạng .npy (NumPy array file) — một định dạng phổ biến trong khoa học dữ liệu, giúp lưu trữ mảng nhiều chiều với tốc độ truy xuất cao, đồng thời dễ dàng nạp lại trong quá trình huấn luyện mô hình.
Sau khi hoàn tất quá trình thu thập, dữ liệu được sắp xếp trong thư mục data/ theo cấu trúc:
