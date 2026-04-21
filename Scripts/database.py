import sqlite3
import hashlib
from datetime import datetime


class DatabaseManager:
    def __init__(self, db_name="medical_system.db"):
        self.db_name = db_name
        self.create_tables()
        self.create_default_admin()  # Tự động kiểm tra và tạo admin khi khởi chạy

    def get_connection(self):
        return sqlite3.connect(self.db_name)

    def hash_password(self, password):
        """Mã hóa mật khẩu bằng SHA-256 để bảo mật"""
        return hashlib.sha256(password.encode()).hexdigest()

    def create_tables(self):
        """Tạo các bảng cần thiết nếu chưa tồn tại"""
        conn = self.get_connection()
        cursor = conn.cursor()

        # 1. Bảng Users
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                full_name TEXT,
                role TEXT, -- 'admin', 'doctor', 'patient'
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # 2. Bảng Lịch sử Nhận diện AI
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS recognition_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                action_name TEXT,
                confidence REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # 3. Bảng Lịch sử Chat
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sender_name TEXT,
                message TEXT,
                message_type TEXT, -- 'manual', 'ai_generated', 'video_sign'
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        conn.commit()
        conn.close()
        print(">> Database & Tables đã sẵn sàng!")

    def create_default_admin(self):
        """Tạo tài khoản admin mặc định nếu chưa có (admin/123456)"""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT * FROM users WHERE username = 'admin'")
            if not cursor.fetchone():
                hashed_pw = self.hash_password("123456")
                cursor.execute('''
                    INSERT INTO users (username, password, full_name, role) 
                    VALUES (?, ?, ?, ?)
                ''', ("admin", hashed_pw, "Quản trị viên Hệ thống", "admin"))
                conn.commit()
                print(">> Đã khởi tạo tài khoản Admin mặc định (admin/123456)")
        except Exception as e:
            print(f"Lỗi khởi tạo admin: {e}")
        finally:
            conn.close()

    # --- CÁC HÀM QUẢN LÝ TÀI KHOẢN ---

    def register_user(self, username, password, full_name, role="patient"):
        """Đăng ký tài khoản mới (Mặc định là patient)"""
        conn = self.get_connection()
        cursor = conn.cursor()

        # Mã hóa mật khẩu trước khi lưu
        hashed_pw = self.hash_password(password)

        try:
            cursor.execute('''
                INSERT INTO users (username, password, full_name, role) 
                VALUES (?, ?, ?, ?)
            ''', (username, hashed_pw, full_name, role))
            conn.commit()
            return True, "Đăng ký thành công!"
        except sqlite3.IntegrityError:
            return False, "Tên đăng nhập đã tồn tại!"
        except Exception as e:
            return False, f"Lỗi không xác định: {e}"
        finally:
            conn.close()

    def login_user(self, username, password):
        """Kiểm tra đăng nhập"""
        conn = self.get_connection()
        cursor = conn.cursor()

        hashed_pw = self.hash_password(password)

        cursor.execute('''
            SELECT role, full_name FROM users 
            WHERE username = ? AND password = ?
        ''', (username, hashed_pw))

        user = cursor.fetchone()
        conn.close()

        if user:
            # Trả về (True, Role, FullName)
            return True, user[0], user[1]
        return False, None, None

    # --- CÁC HÀM QUẢN LÝ DÀNH CHO ADMIN (MỚI THÊM) ---

    def get_all_users(self):
        """Lấy danh sách user (trừ admin) để hiển thị Dashboard"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, username, full_name, role FROM users ORDER BY id DESC")
        data = cursor.fetchall()
        conn.close()
        return data

    def update_user_role(self, user_id, new_role):
        """Cập nhật quyền (role) cho user"""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("UPDATE users SET role = ? WHERE id = ?", (new_role, user_id))
            conn.commit()
            return True
        except:
            return False
        finally:
            conn.close()

    def delete_user(self, user_id):
        """Xóa user khỏi hệ thống"""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
            conn.commit()
        except:
            pass
        finally:
            conn.close()

    # --- CÁC HÀM GHI LOG HỆ THỐNG ---

    def log_recognition(self, action, confidence):
        """Lưu lại khi AI nhận diện được cử chỉ"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('INSERT INTO recognition_logs (action_name, confidence) VALUES (?, ?)',
                       (action, confidence))
        conn.commit()
        conn.close()

    def log_chat(self, sender, message, msg_type="manual"):
        """Lưu lại tin nhắn chat"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('INSERT INTO chat_history (sender_name, message, message_type) VALUES (?, ?, ?)',
                       (sender, message, msg_type))
        conn.commit()
        conn.close()

    # --- CÁC HÀM BÁO CÁO ---

    def get_recent_actions(self, limit=10):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT action_name, confidence, timestamp FROM recognition_logs ORDER BY id DESC LIMIT ?',
                       (limit,))
        data = cursor.fetchall()
        conn.close()
        return data

    def get_user_count(self):
        """Đếm số lượng user theo quyền"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT role, COUNT(*) FROM users GROUP BY role')
        data = cursor.fetchall()
        conn.close()
        return data