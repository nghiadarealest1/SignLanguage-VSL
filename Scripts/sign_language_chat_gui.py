import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox, filedialog
import cv2
from PIL import Image, ImageFont, ImageDraw  # Thêm ImageFont và ImageDraw để vẽ chữ Việt
import numpy as np
import tensorflow as tf
import mediapipe as mp
import threading
import time
import os
import socket
import struct
from datetime import datetime
from vidgear.gears import NetGear
from database import DatabaseManager
import warnings
import speech_recognition as sr

# Tắt cảnh báo Protobuf & Tensorflow
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ================= CẤU HÌNH GIAO DIỆN =================
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")
ctk.set_widget_scaling(1.1)

THEME = {
    "bg_main": "#1e1e2e", "bg_sec": "#252538", "bg_card": "#313244",
    "accent": "#89b4fa", "primary": "#74c7ec", "success": "#a6e3a1",
    "warning": "#f9e2af", "danger": "#f38ba8", "text": "#cdd6f4",
    "text_dim": "#a6adc8", "border": "#45475a",
    "chat_me": "#3b82f6", "chat_other": "#4b5563"
}

FONT_HEADER = ("Roboto", 30, "bold")
FONT_SUBHEADER = ("Roboto", 24, "bold")
FONT_BODY = ("Roboto", 16)
FONT_BOLD_BODY = ("Roboto", 16, "bold")
FONT_ICON = ("Arial", 30)
FONT_HUGE = ("Arial", 50, "bold")

# ================= CẤU HÌNH AI =================
SEQUENCE_LENGTH = 100
THRESHOLD = 0.80

QUICK_RESPONSES = {
    "Ăn đủ": "../responses/an_du.mp4", "Ăn ít": "../responses/an_it.mp4",
    "Bác sĩ": "../responses/bac_si.mp4", "Bao nhiêu?": "../responses/bao_nhieu.mp4",
    "Bệnh phổi": "../responses/benh_phoi.mp4", "Cấp cứu": "../responses/cap_cuu.mp4",
    "Còi xương": "../responses/coi_xuong.mp4", "Đau": "../responses/dau.mp4",
    "Đau bụng": "../responses/dau_bung.mp4", "Đau chân": "../responses/dau_chan.mp4",
    "Đau lưng": "../responses/dau_lung.mp4", "Dị ứng": "../responses/di_ung.mp4",
    "Đột quỵ": "../responses/dot_quy.mp4", "Gan nhễm mỡ": "../responses/gan_nhiem_mo.mp4",
    "Hệ bài tiết": "../responses/he_bai_tiet.mp4", "Hệ miễn dịch": "../responses/he_mien_dich.mp4",
    "Hệ thần kinh": "../responses/he_than_kinh.mp4", "Khám bệnh": "../responses/kham_benh.mp4",
    "Khi nào": "../responses/khi_nao.mp4", "Khỏe mạnh": "../responses/khoe_manh.mp4",
    "Khỏi bệnh": "../responses/khoi_benh.mp4", "Lạnh": "../responses/lanh.mp4",
    "Lây bệnh": "../responses/lay_benh.mp4", "Mang thai": "../responses/mang_thai.mp4",
    "Máy trợ thính": "../responses/may_tro_thinh.mp4", "Mệt": "../responses/met.mp4",
    "Ngất xỉu": "../responses/ngat_xiu.mp4", "Phổi": "../responses/phoi.mp4",
    "Rau": "../responses/rau.mp4", "Sổ mũi": "../responses/so_mui.mp4",
    "Tạm biệt": "../responses/tam_biet.mp4", "Thế nào": "../responses/the_nao.mp4",
    "Thuốc bắc": "../responses/thuoc_bac.mp4", "Thuốc bổ": "../responses/thuoc_bo.mp4",
    "Thuốc lá": "../responses/thuoc_la.mp4", "Uống": "../responses/uong.mp4",
}


# ================= CLASS XỬ LÝ NGÔN NGỮ (OFFLINE) =================
class SmartNLP:
    def __init__(self):
        self.scenarios = [
            # --- NHÓM 1: THỦ TỤC & ĐĂNG KÝ ---
            {"keywords": ["toi_den_kham"], "sentence": "Tôi đến để khám bệnh."},
            {"keywords": ["toi_co_lich"], "sentence": "Tôi đã có lịch hẹn từ trước."},
            {"keywords": ["lay_so"], "sentence": "Tôi muốn lấy số thứ tự."},
            {"keywords": ["lay_so", "o_dau"], "sentence": "Lấy số thứ tự ở đâu ạ?"},
            {"keywords": ["kiem_tra", "o_dau"], "sentence": "Phòng kiểm tra ở đâu vậy?"},
            {"keywords": ["kiem_tra", "suc_khoe"], "sentence": "Tôi muốn đăng ký kiểm tra sức khỏe tổng quát."},

            # --- NHÓM 2: TRIỆU CHỨNG SỐT & MỆT MỎI ---
            {"keywords": ["sot", "nhieu"], "sentence": "Tôi bị sốt rất cao."},
            {"keywords": ["sot", "lau_roi"], "sentence": "Tôi đã bị sốt nhiều ngày nay rồi."},
            {"keywords": ["sot", "met"], "sentence": "Tôi bị sốt và cảm thấy rất mệt."},
            {"keywords": ["khong", "sot"], "sentence": "Tôi không bị sốt."},
            {"keywords": ["met", "nhieu"], "sentence": "Cơ thể tôi cảm thấy rất mệt mỏi."},
            {"keywords": ["chong_mat", "buon_non"], "sentence": "Tôi bị chóng mặt và buồn nôn."},
            {"keywords": ["chong_mat", "nhieu"], "sentence": "Tôi bị chóng mặt dữ dội."},

            # --- NHÓM 3: CÁC VỊ TRÍ ĐAU ---
            {"keywords": ["dau_bung", "nhieu"], "sentence": "Tôi bị đau bụng dữ dội."},
            {"keywords": ["dau_bung", "lau_roi"], "sentence": "Tôi bị đau bụng âm ỉ đã lâu."},
            {"keywords": ["dau_dau", "nhieu"], "sentence": "Tôi bị đau đầu rất nhiều."},
            # Lưu ý: Kiểm tra lại nếu bạn có nhãn 'dau_dau', nếu không thì dùng 'dau_mat'
            {"keywords": ["dau_mat", "do"], "sentence": "Mắt tôi bị đau và đỏ."},
            {"keywords": ["dau_tay"], "sentence": "Tôi bị đau ở vùng tay."},
            {"keywords": ["dau_chan", "nhieu"], "sentence": "Chân tôi đau quá, không đi lại được."},

            # --- NHÓM 4: HÔ HẤP & DỊ ỨNG ---
            {"keywords": ["ho", "nhieu"], "sentence": "Tôi bị ho rất nhiều."},
            {"keywords": ["ho", "lau_roi"], "sentence": "Tôi bị ho kéo dài đã lâu không khỏi."},
            {"keywords": ["kho_tho", "nhieu"], "sentence": "Tôi cảm thấy rất khó thở."},
            {"keywords": ["khong", "kho_tho"], "sentence": "Tôi thở bình thường, không bị khó thở."},
            {"keywords": ["di_ung", "thuoc"], "sentence": "Tôi có tiền sử dị ứng với thuốc."},
            {"keywords": ["di_ung", "do"], "sentence": "Da tôi bị dị ứng nổi mẩn đỏ."},

            # --- NHÓM 5: THAI SẢN ---
            {"keywords": ["mang_thai", "kiem_tra"], "sentence": "Tôi đến khám thai định kỳ."},
            {"keywords": ["mang_thai", "met"], "sentence": "Tôi đang mang thai và thấy rất mệt."},
            {"keywords": ["mang_thai", "dau_bung"], "sentence": "Tôi đang mang thai nhưng bị đau bụng, tôi rất lo."},

            # --- NHÓM 6: THUỐC & ĐIỀU TRỊ ---
            {"keywords": ["thuoc", "uong", "khi_nao"], "sentence": "Thuốc này uống vào lúc nào?"},
            {"keywords": ["thuoc", "uong", "the_nao"], "sentence": "Thuốc này uống như thế nào (trước hay sau ăn)?"},
            {"keywords": ["quen", "uong", "thuoc"], "sentence": "Tôi bị quên uống thuốc."},
            {"keywords": ["khong", "uong", "thuoc"], "sentence": "Tôi chưa uống thuốc."},

            # --- NHÓM 7: GIAO TIẾP KHÁC ---
            {"keywords": ["suc_khoe", "the_nao"], "sentence": "Tình hình sức khỏe của tôi thế nào bác sĩ?"},
            {"keywords": ["toi_hieu_roi"], "sentence": "Vâng, tôi đã hiểu rồi ạ."},
            {"keywords": ["khong", "hieu"], "sentence": "Tôi vẫn chưa hiểu rõ lắm."},
        ]

    def process(self, word_list):
        if not word_list: return ""
        input_lower = [w.lower() for w in word_list]
        best_match_sentence = None
        max_matches = 0

        for scenario in self.scenarios:
            matches = 0
            for kw in scenario["keywords"]:
                if any(kw in word for word in input_lower):
                    matches += 1

            if (matches >= 2 and matches > max_matches) or \
                    (matches == 1 and len(scenario["keywords"]) == 1 and max_matches == 0):
                max_matches = matches
                best_match_sentence = scenario["sentence"]

        if best_match_sentence:
            return best_match_sentence
        return " ".join(word_list)


# ================= HÀM HỖ TRỢ =================
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z] for res in
                     results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 3)
    lh = np.array([[res.x, res.y, res.z] for res in
                   results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in
                   results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(
        21 * 3)
    return np.concatenate([pose, lh, rh])


# ================= CLASS LOGIN =================
class LoginApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Hệ thống Y tế - Đăng nhập")
        self.geometry("500x650")
        self.db = DatabaseManager()
        self.configure(fg_color=THEME["bg_main"])
        self.resizable(False, False)

        self.tab_view = ctk.CTkTabview(self, width=450, height=600, fg_color=THEME["bg_sec"])
        self.tab_view.pack(padx=20, pady=20)
        self.tab_view._segmented_button.configure(font=FONT_BOLD_BODY)
        self.tab_view.add("Đăng nhập")
        self.tab_view.add("Đăng ký")
        self.setup_login_tab()
        self.setup_register_tab()

    def setup_login_tab(self):
        tab = self.tab_view.tab("Đăng nhập")
        ctk.CTkLabel(tab, text="Đăng nhập hệ thống", font=FONT_HEADER, text_color=THEME["primary"]).pack(pady=40)
        self.entry_user_login = ctk.CTkEntry(tab, placeholder_text="Tên đăng nhập", width=350, height=50,
                                             font=FONT_BODY)
        self.entry_user_login.pack(pady=15)
        self.entry_pass_login = ctk.CTkEntry(tab, placeholder_text="Mật khẩu", show="*", width=350, height=50,
                                             font=FONT_BODY)
        self.entry_pass_login.pack(pady=15)
        ctk.CTkButton(tab, text="ĐĂNG NHẬP", width=350, height=55, fg_color=THEME["primary"], font=FONT_BOLD_BODY,
                      command=self.login).pack(pady=30)

    def setup_register_tab(self):
        tab = self.tab_view.tab("Đăng ký")
        ctk.CTkLabel(tab, text="Đăng ký tài khoản", font=FONT_HEADER, text_color=THEME["success"]).pack(pady=30)
        self.entry_user_reg = ctk.CTkEntry(tab, placeholder_text="Tên đăng nhập", width=350, height=45, font=FONT_BODY)
        self.entry_user_reg.pack(pady=10)
        self.entry_pass_reg = ctk.CTkEntry(tab, placeholder_text="Mật khẩu", show="*", width=350, height=45,
                                           font=FONT_BODY)
        self.entry_pass_reg.pack(pady=10)
        self.entry_name_reg = ctk.CTkEntry(tab, placeholder_text="Họ và tên hiển thị", width=350, height=45,
                                           font=FONT_BODY)
        self.entry_name_reg.pack(pady=10)
        ctk.CTkLabel(tab, text="* Tài khoản mới mặc định là Bệnh nhân.", text_color="gray", font=("Arial", 14)).pack(
            pady=10)
        ctk.CTkButton(tab, text="ĐĂNG KÝ NGAY", width=350, height=50, fg_color=THEME["success"], text_color="black",
                      font=FONT_BOLD_BODY, command=self.register).pack(pady=20)

    def login(self):
        u = self.entry_user_login.get().strip()
        p = self.entry_pass_login.get().strip()
        success, role, fullname = self.db.login_user(u, p)
        if success:
            self.destroy()
            app = GraduationProjectApp(user_role=role, user_name=fullname)
            app.mainloop()
        else:
            messagebox.showerror("Thất bại", "Sai tài khoản hoặc mật khẩu!")

    def register(self):
        u = self.entry_user_reg.get().strip()
        p = self.entry_pass_reg.get().strip()
        n = self.entry_name_reg.get().strip()
        if not u or not p or not n:
            messagebox.showwarning("Thiếu thông tin", "Vui lòng nhập đủ thông tin!")
            return
        success, msg = self.db.register_user(u, p, n, role="patient")
        if success:
            messagebox.showinfo("Thành công", msg)
            self.tab_view.set("Đăng nhập")
        else:
            messagebox.showerror("Lỗi", msg)


# ================= CLASS APP CHÍNH =================
class GraduationProjectApp(ctk.CTk):
    def __init__(self, user_role, user_name):
        super().__init__()
        self.role = user_role
        self.username = user_name
        self.title(f"Hệ thống Y tế Thông minh - {user_name} ({user_role.upper()})")
        self.geometry("1366x768")
        self.minsize(1200, 800)
        self.configure(fg_color=THEME["bg_main"])

        self.db = DatabaseManager()
        self.nlp_engine = SmartNLP()

        # Config AI
        self.sequence = []
        self.is_calling = False
        self.client_sender = None
        self.server_receiver = None
        self.response_cap = None
        self.is_playing_response = False
        self.model = None

        self.sentence = []
        self.predictions = []
        self.last_prediction_time = time.time()

        # Threading
        self.thread_running = True
        self.latest_frame = None
        self.frame_lock = threading.Lock()

        # File Transfer
        self.file_port = 5555
        self.file_save_dir = "../Hinh_anh_nhan_duoc"
        if not os.path.exists(self.file_save_dir): os.makedirs(self.file_save_dir)
        threading.Thread(target=self.start_file_server, daemon=True).start()

        # Speech
        self.recognizer = sr.Recognizer()
        self.is_listening = False

        if self.role == "patient":
            self.load_ai_model()
            self.mp_holistic = mp.solutions.holistic
            self.holistic = self.mp_holistic.Holistic(
                min_detection_confidence=0.5, min_tracking_confidence=0.5,
                model_complexity=0, smooth_landmarks=True
            )
            self.mp_drawing = mp.solutions.drawing_utils

        if self.role != "admin":
            self.cap = cv2.VideoCapture(0)
            self.cap.set(3, 640)
            self.cap.set(4, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
        else:
            self.cap = None

        self.setup_ui()
        if self.cap:
            self.video_thread = threading.Thread(target=self.video_processing_loop, daemon=True)
            self.video_thread.start()
            self.update_ui_loop()

        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def load_ai_model(self):
        try:
            self.model = tf.keras.models.load_model("../Models/gesture_model.keras")
            self.label_map = np.load("../Models/label_map.npy", allow_pickle=True).item()
        except:
            self.model = None

    # ================= HÀM HỖ TRỢ VẼ TIẾNG VIỆT =================
    def draw_vietnamese_text(self, img_array, text, pos, font_size, color):
        """Hỗ trợ vẽ tiếng Việt lên ảnh OpenCV thông qua PIL"""
        try:
            # 1. Chuyển từ OpenCV (BGR) sang PIL
            pil_img = Image.fromarray(img_array)
            draw = ImageDraw.Draw(pil_img)

            # 2. Thử load font Arial (Windows)
            try:
                # Bạn có thể thay đường dẫn font nếu muốn (vd: "arial.ttf" hoặc đường dẫn tuyệt đối)
                font = ImageFont.truetype("arial.ttf", font_size)
            except:
                # Nếu không tìm thấy font, dùng font mặc định (sẽ không đẹp bằng)
                font = ImageFont.load_default()

            # 3. Vẽ chữ
            draw.text(pos, text, font=font, fill=color)

            # 4. Chuyển ngược lại về Numpy array
            return np.array(pil_img)
        except Exception as e:
            print(f"Font Error: {e}")
            return img_array

    def setup_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        # Sidebar
        self.sidebar = ctk.CTkFrame(self, width=100, corner_radius=0, fg_color=THEME["bg_sec"])
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        icon = "🛠️" if self.role == "admin" else "👨‍⚕️" if self.role == "doctor" else "🤕"
        ctk.CTkLabel(self.sidebar, text=icon, font=("Arial", 48)).pack(pady=40)
        self.nav_btns = {}
        self.create_nav_btn("dashboard", "📊", self.show_dashboard)
        if self.role != "admin":
            self.create_nav_btn("meeting", "📹", self.show_meeting)
            self.create_nav_btn("chat", "💬", self.show_chat)
        if self.role == "doctor":
            ctk.CTkButton(self.sidebar, text="💾", width=60, height=60, fg_color="transparent", font=("Arial", 30),
                          hover_color=THEME["success"], command=self.export_medical_record).pack(pady=15)
        ctk.CTkButton(self.sidebar, text="🚪", width=60, height=60, fg_color="transparent", font=("Arial", 30),
                      hover_color=THEME["danger"], command=self.logout).pack(side="bottom", pady=30)
        # Main
        self.main_area = ctk.CTkFrame(self, corner_radius=20, fg_color=THEME["bg_main"])
        self.main_area.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.main_area.grid_rowconfigure(0, weight=1)
        self.main_area.grid_columnconfigure(0, weight=1)
        self.frames = {}
        pages = ["dashboard"]
        if self.role != "admin": pages.extend(["meeting", "chat"])
        for page in pages:
            fr = ctk.CTkFrame(self.main_area, fg_color="transparent")
            fr.grid(row=0, column=0, sticky="nsew")
            self.frames[page] = fr
        self.build_dashboard_ui()
        if self.role != "admin":
            self.build_meeting_ui()
            self.build_chat_ui()
        self.switch_tab("dashboard")

    def create_nav_btn(self, name, icon, command):
        btn = ctk.CTkButton(self.sidebar, text=icon, width=60, height=60, corner_radius=15,
                            fg_color="transparent", hover_color=THEME["accent"], font=("Arial", 30), command=command)
        btn.pack(pady=15)
        self.nav_btns[name] = btn

    def switch_tab(self, page_name):
        for name, frame in self.frames.items():
            if name == page_name: frame.tkraise()
        for name, btn in self.nav_btns.items():
            if btn:
                btn.configure(fg_color=THEME["primary"] if name == page_name else "transparent",
                              text_color=THEME["bg_main"] if name == page_name else THEME["text"])

    def show_dashboard(self):
        self.switch_tab("dashboard")

    def show_meeting(self):
        self.switch_tab("meeting")

    def show_chat(self):
        self.switch_tab("chat")

    def logout(self):
        self.on_closing()

    def export_medical_record(self):
        messagebox.showinfo("Thông báo", "Vui lòng xem log chat trong Database.")

    # ================= UI DASHBOARD =================
    def build_dashboard_ui(self):
        f = self.frames["dashboard"]
        header = ctk.CTkFrame(f, fg_color="transparent")
        header.pack(fill="x", padx=30, pady=(30, 20))
        ctk.CTkLabel(header, text=f"Xin chào, {self.username}", font=FONT_HEADER).pack(side="left")
        if self.role == "admin":
            ctk.CTkButton(header, text="🔄 Làm mới danh sách", width=180, height=45, font=FONT_BOLD_BODY,
                          command=self.refresh_user_list).pack(side="right")
        grid = ctk.CTkFrame(f, fg_color="transparent")
        grid.pack(fill="x", padx=30)
        self.create_stat_card(grid, "Vai trò", self.role.upper(), THEME["primary"], 0)
        if self.role == "admin":
            ctk.CTkLabel(f, text="QUẢN LÝ NGƯỜI DÙNG", font=FONT_SUBHEADER, text_color=THEME["accent"]).pack(anchor="w",
                                                                                                             padx=30,
                                                                                                             pady=(30,
                                                                                                                   15))
            list_header = ctk.CTkFrame(f, height=50, fg_color=THEME["bg_sec"])
            list_header.pack(fill="x", padx=30)
            columns = ["ID", "Tên đăng nhập", "Họ tên", "Quyền", "Hành động"]
            for i in range(5): list_header.grid_columnconfigure(i, weight=1)
            for i, col in enumerate(columns):
                ctk.CTkLabel(list_header, text=col, font=("Arial", 16, "bold")).grid(row=0, column=i, sticky="ew",
                                                                                     padx=5, pady=10)
            self.user_list_frame = ctk.CTkScrollableFrame(f, fg_color="transparent")
            self.user_list_frame.pack(fill="both", expand=True, padx=30, pady=10)
            self.refresh_user_list()
        else:
            self.create_stat_card(grid, "Trạng thái", "Sẵn sàng", THEME["success"], 1)

    def create_stat_card(self, parent, title, value, color, col_idx):
        card = ctk.CTkFrame(parent, fg_color=THEME["bg_card"], corner_radius=15)
        card.grid(row=0, column=col_idx, padx=10, pady=10, sticky="ew")
        parent.grid_columnconfigure(col_idx, weight=1)
        ctk.CTkLabel(card, text=title, font=FONT_BODY, text_color=THEME["text_dim"]).pack(anchor="w", padx=25,
                                                                                          pady=(20, 5))
        ctk.CTkLabel(card, text=value, font=("Roboto", 36, "bold"), text_color=color).pack(anchor="w", padx=25,
                                                                                           pady=(0, 20))

    def refresh_user_list(self):
        for widget in self.user_list_frame.winfo_children(): widget.destroy()
        users = self.db.get_all_users()
        for i in range(5): self.user_list_frame.grid_columnconfigure(i, weight=1)
        for idx, u in enumerate(users):
            uid, uname, fname, urole = u
            if uname == "admin": continue
            ctk.CTkLabel(self.user_list_frame, text=str(uid), font=FONT_BODY).grid(row=idx, column=0, pady=10)
            ctk.CTkLabel(self.user_list_frame, text=uname, font=FONT_BODY).grid(row=idx, column=1, pady=10)
            ctk.CTkLabel(self.user_list_frame, text=fname, font=FONT_BODY).grid(row=idx, column=2, pady=10)
            role_color = THEME["primary"] if urole == "doctor" else THEME["warning"]
            ctk.CTkLabel(self.user_list_frame, text=urole.upper(), text_color=role_color, font=FONT_BOLD_BODY).grid(
                row=idx, column=3, pady=10)
            action_frame = ctk.CTkFrame(self.user_list_frame, fg_color="transparent")
            action_frame.grid(row=idx, column=4, pady=10)
            if urole == "patient":
                ctk.CTkButton(action_frame, text="🔼 Lên Bác sĩ", width=120, height=35, font=FONT_BODY,
                              fg_color=THEME["success"], command=lambda i=uid: self.change_role(i, "doctor")).pack(
                    side="left", padx=5)
            elif urole == "doctor":
                ctk.CTkButton(action_frame, text="🔽 Xuống BN", width=120, height=35, font=FONT_BODY,
                              fg_color=THEME["warning"], command=lambda i=uid: self.change_role(i, "patient")).pack(
                    side="left", padx=5)
            ctk.CTkButton(action_frame, text="🗑️ Xóa", width=80, height=35, font=FONT_BODY, fg_color=THEME["danger"],
                          command=lambda i=uid: self.delete_user(i)).pack(side="left", padx=5)

    def change_role(self, user_id, new_role):
        if self.db.update_user_role(user_id, new_role): self.refresh_user_list()

    def delete_user(self, user_id):
        if messagebox.askyesno("Xác nhận", "Xóa user này?"): self.db.delete_user(user_id); self.refresh_user_list()

    # ================= UI MEETING =================
    def build_meeting_ui(self):
        f = self.frames["meeting"]
        f.grid_columnconfigure(0, weight=3)
        f.grid_columnconfigure(1, weight=1)
        f.grid_rowconfigure(0, weight=1)
        if self.role == "patient":
            lbl_local, lbl_remote = "Bệnh nhân (Bạn)", "Bác sĩ"
            col_local, col_remote = THEME["warning"], THEME["primary"]
            ai_text, ip_holder = "AI ĐANG DỊCH...", "Nhập IP Bác sĩ..."
        else:
            lbl_local, lbl_remote = "Bác sĩ (Bạn)", "Bệnh nhân"
            col_local, col_remote = THEME["primary"], THEME["warning"]
            ai_text, ip_holder = "DỮ LIỆU TỪ BỆNH NHÂN", "Nhập IP Bệnh nhân..."

        video_panel = ctk.CTkFrame(f, fg_color="transparent")
        video_panel.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.ai_banner = ctk.CTkFrame(video_panel, height=80, fg_color=THEME["bg_card"], corner_radius=15)
        self.ai_banner.pack(fill="x", pady=(0, 15))
        ctk.CTkLabel(self.ai_banner, text=ai_text, font=("Arial", 18, "bold"), text_color=col_local).pack(side="left",
                                                                                                          padx=25)
        self.trans_text = ctk.CTkLabel(self.ai_banner, text="...", font=("Arial", 32, "bold"), text_color=THEME["text"])
        self.trans_text.pack(side="left", padx=15)

        cam_grid = ctk.CTkFrame(video_panel, fg_color="transparent")
        cam_grid.pack(fill="both", expand=True)
        cam_grid.grid_columnconfigure(0, weight=1)
        cam_grid.grid_columnconfigure(1, weight=1)
        cam_grid.grid_rowconfigure(0, weight=1)

        self.local_cam_frame = ctk.CTkFrame(cam_grid, fg_color="black", corner_radius=15, border_width=4,
                                            border_color=col_local)
        self.local_cam_frame.grid(row=0, column=0, sticky="nsew", padx=5)
        ctk.CTkLabel(self.local_cam_frame, text=lbl_local, fg_color=col_local, text_color="black", font=FONT_BOLD_BODY,
                     corner_radius=8).place(x=10, y=10)
        self.local_cam_label = ctk.CTkLabel(self.local_cam_frame, text="")
        self.local_cam_label.pack(fill="both", expand=True, padx=2, pady=2)

        self.remote_cam_frame = ctk.CTkFrame(cam_grid, fg_color="black", corner_radius=15, border_width=2,
                                             border_color=col_remote)
        self.remote_cam_frame.grid(row=0, column=1, sticky="nsew", padx=5)
        ctk.CTkLabel(self.remote_cam_frame, text=lbl_remote, fg_color=col_remote, text_color="black",
                     font=FONT_BOLD_BODY, corner_radius=8).place(x=10, y=10)
        self.remote_cam_label = ctk.CTkLabel(self.remote_cam_frame, text="Đang chờ kết nối...", text_color="gray",
                                             font=FONT_HEADER)
        self.remote_cam_label.pack(fill="both", expand=True, padx=2, pady=2)

        # Subtitle Label
        self.subtitle_label = ctk.CTkLabel(self.remote_cam_frame, text="", font=FONT_HUGE, text_color="yellow",
                                           bg_color="transparent")
        self.subtitle_label.place(relx=0.5, rely=0.85, anchor="center")

        ctrl = ctk.CTkFrame(f, fg_color=THEME["bg_sec"], corner_radius=20)
        ctrl.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        ctk.CTkLabel(ctrl, text="Kết nối Video", font=FONT_HEADER).pack(pady=25)
        self.ip_entry = ctk.CTkEntry(ctrl, placeholder_text=ip_holder, height=45, font=FONT_BODY)
        self.ip_entry.pack(fill="x", padx=15, pady=10)
        self.btn_call = ctk.CTkButton(ctrl, text="📞 Gọi", fg_color=THEME["success"], height=50, font=FONT_BOLD_BODY,
                                      command=self.start_video_call)
        self.btn_call.pack(fill="x", padx=15, pady=10)
        self.btn_stop = ctk.CTkButton(ctrl, text="❌ Ngắt", fg_color=THEME["danger"], height=50, font=FONT_BOLD_BODY,
                                      state="disabled", command=self.stop_video_call)
        self.btn_stop.pack(fill="x", padx=15, pady=10)

        ctk.CTkLabel(ctrl, text="Mẫu câu nhanh", font=FONT_SUBHEADER, text_color=THEME["primary"]).pack(anchor="w",
                                                                                                        padx=20,
                                                                                                        pady=(30, 10))
        scroll = ctk.CTkScrollableFrame(ctrl, fg_color="transparent")
        scroll.pack(fill="both", expand=True, padx=5, pady=5)
        for txt, path in QUICK_RESPONSES.items():
            ctk.CTkButton(scroll, text=f"🎬 {txt}", anchor="w", fg_color=THEME["bg_card"], height=40, font=FONT_BODY,
                          command=lambda p=path, t=txt: self.trigger_video_response(p, t)).pack(fill="x", pady=5)

    # ================= UI CHAT =================
    def build_chat_ui(self):
        f = self.frames["chat"]
        f.grid_columnconfigure(0, weight=1)
        f.grid_rowconfigure(0, weight=1)
        f.grid_rowconfigure(1, weight=0)
        self.chat_scroll = ctk.CTkScrollableFrame(f, fg_color=THEME["bg_card"])
        self.chat_scroll.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
        inp = ctk.CTkFrame(f, fg_color="transparent")
        inp.grid(row=1, column=0, sticky="ew", padx=20, pady=(0, 20))
        self.btn_img = ctk.CTkButton(inp, text="📷 Ảnh", width=80, height=60, font=FONT_BOLD_BODY,
                                     fg_color=THEME["accent"], text_color="black", command=self.select_and_send_image)
        self.btn_img.pack(side="left", padx=(0, 5))
        self.msg_entry = ctk.CTkEntry(inp, placeholder_text="Nhập tin nhắn...", height=60, corner_radius=25,
                                      font=("Roboto", 22))
        self.msg_entry.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self.msg_entry.bind("<Return>", self.send_message_manual)
        if self.role == "doctor":
            self.btn_mic = ctk.CTkButton(inp, text="🎤 Nói", width=80, height=60, font=FONT_BOLD_BODY, corner_radius=25,
                                         fg_color=THEME["primary"], command=self.toggle_speech)
            self.btn_mic.pack(side="right", padx=5)
        ctk.CTkButton(inp, text="Gửi", width=120, height=60, corner_radius=25, font=FONT_BOLD_BODY,
                      command=self.send_message_manual).pack(side="right")

    def add_message_to_ui(self, sender, content, color, is_image=False):
        align = "e" if "Tôi" in sender or sender == "Me" else "w"
        bubble_bg = THEME["chat_me"] if align == "e" else THEME["chat_other"]
        sender_text = "" if align == "e" else f"{sender} • {datetime.now().strftime('%H:%M')}"

        container = ctk.CTkFrame(self.chat_scroll, fg_color="transparent")
        container.pack(fill="x", pady=5, padx=10)
        if sender_text: ctk.CTkLabel(container, text=sender_text, font=("Arial", 12), text_color="gray").pack(
            anchor=align, padx=5)

        if is_image:
            try:
                pil_img = Image.open(content)
                ratio = 300 / float(pil_img.size[0])
                img_ctk = ctk.CTkImage(light_image=pil_img, size=(300, int(float(pil_img.size[1]) * float(ratio))))
                msg = ctk.CTkLabel(container, text="", image=img_ctk)
                msg.bind("<Button-1>", lambda e, p=content: self.open_image_viewer(p))
            except:
                msg = ctk.CTkLabel(container, text="[Lỗi ảnh]", text_color="red")
        else:
            msg = ctk.CTkLabel(container, text=content, font=("Roboto", 20), fg_color=bubble_bg, text_color="white",
                               corner_radius=15, padx=15, pady=10, wraplength=450, justify="left")

        msg.pack(anchor=align)
        self.chat_scroll.update_idletasks()
        self.chat_scroll._parent_canvas.yview_moveto(1.0)
        if not is_image and hasattr(self, 'subtitle_label'):
            self.subtitle_label.configure(text=content)
            self.after(5000, lambda: self.subtitle_label.configure(text=""))

    def open_image_viewer(self, path):
        try:
            os.startfile(path) if os.name == 'nt' else os.system(f"xdg-open '{path}'")
        except:
            pass

    # ================= LOGIC GIAO TIẾP =================
    def start_file_server(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            server.bind(("0.0.0.0", self.file_port))
            server.listen(5)
            while True:
                conn, _ = server.accept()
                threading.Thread(target=self.receive_file_handler, args=(conn,)).start()
        except Exception:
            pass

    def receive_file_handler(self, conn):
        try:
            # 1. Đọc tên file/gói tin
            name_len_data = conn.recv(4)
            if not name_len_data: return
            name_len = struct.unpack("I", name_len_data)[0]
            name = conn.recv(name_len).decode("utf-8")

            # 2. Đọc kích thước dữ liệu
            size_data = conn.recv(8)
            if not size_data: return
            size = struct.unpack("Q", size_data)[0]

            # --- TRƯỜNG HỢP LÀ TIN NHẮN CHAT ---
            if name == "__CHAT__":
                data = b""
                while len(data) < size:
                    chunk = conn.recv(4096)
                    if not chunk: break
                    data += chunk
                text_msg = data.decode("utf-8")
                # Hiển thị lên giao diện (Sử dụng after để thread an toàn với UI)
                self.after(0, lambda: self.add_message_to_ui("ĐỐI PHƯƠNG", text_msg, THEME["chat_other"]))

            # --- TRƯỜNG HỢP LÀ FILE ẢNH ---
            else:
                path = os.path.join(self.file_save_dir, f"recv_{int(time.time())}_{name}")
                recvd = 0
                with open(path, "wb") as f:
                    while recvd < size:
                        chunk = conn.recv(4096)
                        if not chunk: break
                        f.write(chunk)
                        recvd += len(chunk)
                self.after(0, lambda: self.add_message_to_ui("ĐỐI PHƯƠNG", path, THEME["success"], is_image=True))

        except Exception as e:
            print(f"Lỗi nhận dữ liệu: {e}")
        finally:
            conn.close()

    def select_and_send_image(self):
        ip = self.ip_entry.get().strip()
        if not ip: return messagebox.showwarning("Thiếu IP", "Nhập IP đối phương trước!")
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
        if path: threading.Thread(target=self.send_image_thread, args=(ip, path)).start()

    def send_image_thread(self, ip, path):
        try:
            name = os.path.basename(path)
            size = os.path.getsize(path)
            client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client.connect((ip, self.file_port))
            client.sendall(struct.pack("I", len(name)))
            client.sendall(name.encode("utf-8"))
            client.sendall(struct.pack("Q", size))
            with open(path, "rb") as f:
                while True:
                    chunk = f.read(4096)
                    if not chunk: break
                    client.sendall(chunk)
            client.close()
            self.after(0, lambda: self.add_message_to_ui("Tôi", path, THEME["accent"], is_image=True))
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không gửi được ảnh: {e}")

    def toggle_speech(self):
        if not self.is_listening:
            self.is_listening = True
            self.btn_mic.configure(fg_color=THEME["danger"], text="...")
            threading.Thread(target=self.listen_thread).start()
        else:
            self.is_listening = False
            self.btn_mic.configure(fg_color=THEME["primary"], text="🎤 Nói")

    def listen_thread(self):
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source)
            try:
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
                text = self.recognizer.recognize_google(audio, language="vi-VN")
                self.msg_entry.delete(0, tk.END);
                self.msg_entry.insert(0, text)
                self.after(0, self.send_message_manual)
            except:
                pass
            finally:
                self.is_listening = False
                self.btn_mic.configure(fg_color=THEME["primary"], text="🎤 Nói")

    def start_video_call(self):
        ip = self.ip_entry.get().strip()
        if not ip: return messagebox.showwarning("Lỗi", "Nhập IP đối phương!")
        try:
            self.client_sender = NetGear(address=ip, port="5454", protocol="tcp", pattern=1, receive_mode=False,
                                         logging=False)
            self.is_calling = True
            threading.Thread(target=self.receive_video_stream, daemon=True).start()
            self.btn_call.configure(state="disabled", text="Đang gọi...")
            self.btn_stop.configure(state="normal")
        except:
            pass

    def stop_video_call(self):
        self.is_calling = False
        if self.client_sender: self.client_sender.close(); self.client_sender = None
        if hasattr(self, 'server_receiver') and self.server_receiver: self.server_receiver.close()
        self.btn_call.configure(state="normal", text="📞 Gọi")
        self.btn_stop.configure(state="disabled")
        self.remote_cam_label.configure(image=None, text="Đã ngắt")

    def receive_video_stream(self):
        try:
            self.server_receiver = NetGear(address="0.0.0.0", port="5454", protocol="tcp", pattern=1, receive_mode=True,
                                           logging=False)
            while self.is_calling:
                frame = self.server_receiver.recv()
                if frame is None: continue
                frame = cv2.resize(frame, (400, 300))
                img = ctk.CTkImage(light_image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), size=(400, 300))
                try:
                    self.remote_cam_label.configure(image=img, text="")
                except:
                    pass
        except:
            pass
        finally:
            if hasattr(self, 'server_receiver') and self.server_receiver: self.server_receiver.close()

    def send_message_manual(self, event=None):
        msg = self.msg_entry.get().strip()
        ip = self.ip_entry.get().strip()  # Lấy IP từ ô nhập liệu

        if msg:
            # 1. Hiển thị lên màn hình mình trước
            self.add_message_to_ui(f"{self.role.capitalize()} (Tôi)", msg, THEME["warning"])
            self.db.log_chat(self.role, msg, "manual")

            # 2. Gửi qua mạng (Nếu có IP)
            if ip:
                threading.Thread(target=self.send_text_thread, args=(ip, msg)).start()

            # 3. Xóa ô nhập
            self.msg_entry.delete(0, tk.END)

    def send_text_thread(self, ip, text):
        try:
            # Quy ước: Tin nhắn chat sẽ có tên file đặc biệt là "__CHAT__"
            name = "__CHAT__"
            data_bytes = text.encode("utf-8")
            size = len(data_bytes)

            client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client.connect((ip, self.file_port))  # Kết nối vào cổng 5555

            # Gửi cấu trúc giống hệt gửi ảnh: [Độ dài tên] [Tên] [Kích thước] [Dữ liệu]
            client.sendall(struct.pack("I", len(name)))
            client.sendall(name.encode("utf-8"))
            client.sendall(struct.pack("Q", size))
            client.sendall(data_bytes)

            client.close()
        except Exception as e:
            print(f"Lỗi gửi tin nhắn: {e}")

    def trigger_video_response(self, path, text):
        if os.path.exists(path):
            self.response_cap = cv2.VideoCapture(path)
            self.is_playing_response = True
            self.add_message_to_ui("Tôi (Video)", text, THEME["accent"])
            self.db.log_chat(self.role, text, "video_sign")

    # ================= LOGIC XỬ LÝ VIDEO & AI =================
    def video_processing_loop(self):
        frame_skip_rate = 5
        frame_counter = 0
        current_text = "..."
        current_conf = 0.0

        while self.thread_running:
            if not self.cap: time.sleep(0.1); continue
            try:
                if self.is_playing_response and self.response_cap:
                    ret, f = self.response_cap.read()
                    if not ret:
                        self.is_playing_response = False;
                        self.response_cap.release();
                        self.response_cap = None
                        _, frame = self.cap.read()
                    else:
                        frame = cv2.resize(f, (640, 480))
                        time.sleep(0.03)  # Giữ tốc độ gốc cho video mẫu
                else:
                    ret, frame = self.cap.read()
            except:
                continue

            if frame is None: continue
            if not self.is_playing_response: frame = cv2.flip(frame, 1)

            display_img = frame.copy()
            display_img_rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)

            if self.role == "patient" and not self.is_playing_response and self.model:
                display_img_rgb.flags.writeable = False
                results = self.holistic.process(display_img_rgb)
                display_img_rgb.flags.writeable = True

                # Chỉ vẽ tay để tối ưu
                self.mp_drawing.draw_landmarks(display_img_rgb, results.left_hand_landmarks,
                                               self.mp_holistic.HAND_CONNECTIONS)
                self.mp_drawing.draw_landmarks(display_img_rgb, results.right_hand_landmarks,
                                               self.mp_holistic.HAND_CONNECTIONS)

                keypoints = extract_keypoints(results)
                self.sequence.append(keypoints)
                if len(self.sequence) > SEQUENCE_LENGTH: self.sequence = self.sequence[-SEQUENCE_LENGTH:]

                # Chỉ dự đoán mỗi 5 frame
                frame_counter += 1
                if len(self.sequence) == SEQUENCE_LENGTH and frame_counter % frame_skip_rate == 0:
                    res = self.model.predict(np.expand_dims(self.sequence, axis=0), verbose=0)[0]
                    idx = np.argmax(res)
                    confidence = res[idx]
                    current_conf = confidence
                    current_text = self.label_map.get(idx, "...")

                    self.predictions.append(idx)
                    if len(self.predictions) > 10: self.predictions = self.predictions[-10:]

                    if np.unique(self.predictions)[-1] == idx:
                        if confidence > 0.85:  # Threshold nhận diện vào câu
                            self.last_prediction_time = time.time()  # Cập nhật thời gian
                            if len(self.sentence) > 0:
                                if current_text != self.sentence[-1]:
                                    self.sentence.append(current_text)
                                    self.db.log_chat(self.role, current_text, "ai_word")
                            else:
                                self.sentence.append(current_text)

                if len(self.sentence) > 5: self.sentence = self.sentence[-5:]

                # Reset câu nếu im lặng quá 5 giây
                if time.time() - self.last_prediction_time > 5.0: self.sentence = []

                # --- VẼ HÌNH (SỬ DỤNG HÀM VẼ CHỮ VIỆT MỚI) ---

                # 1. Chữ xanh (Góc trái):
                if current_conf > 0.80:
                    text_hien_tai = f"Hiện tại: {current_text} ({int(current_conf * 100)}%)"
                    display_img_rgb = self.draw_vietnamese_text(
                        display_img_rgb, text_hien_tai, (10, 10), 24, (0, 255, 0)
                    )

                # 2. Thanh cam (Dưới cùng): Hiện câu đã DỊCH
                final_sentence = self.nlp_engine.process(self.sentence) if self.sentence else "..."

                # Vẽ khối màu cam (dùng cv2 cho nhanh vì là hình khối)
                cv2.rectangle(display_img_rgb, (0, 430), (640, 480), (245, 117, 16), -1)

                # Vẽ chữ tiếng Việt lên khối cam
                display_img_rgb = self.draw_vietnamese_text(
                    display_img_rgb, final_sentence, (10, 440), 28, (255, 255, 255)
                )

            if self.is_calling and self.client_sender:
                try:
                    self.client_sender.send(cv2.cvtColor(display_img_rgb, cv2.COLOR_RGB2BGR))
                except:
                    pass

            with self.frame_lock:
                self.latest_frame = display_img_rgb
            time.sleep(0.001)

    def update_ui_loop(self):
        if hasattr(self, 'frames') and self.frames["meeting"].winfo_ismapped():
            frame = None
            with self.frame_lock:
                if self.latest_frame is not None: frame = self.latest_frame.copy()
            if frame is not None:
                img_tk = ctk.CTkImage(light_image=Image.fromarray(cv2.resize(frame, (480, 360))), size=(480, 360))
                try:
                    self.local_cam_label.configure(image=img_tk)
                except:
                    pass
        self.after(30, self.update_ui_loop)

    def on_closing(self):
        self.thread_running = False
        try:
            self.is_calling = False;
            self.is_listening = False
            if self.cap: self.cap.release()
            if self.client_sender: self.client_sender.close()
            if hasattr(self, 'server_receiver') and self.server_receiver: self.server_receiver.close()
        except:
            pass
        self.destroy()
        os._exit(0)


if __name__ == "__main__":
    app = LoginApp()
    app.mainloop()