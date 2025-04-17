import dlib
import numpy as np
import cv2
import os
import pandas as pd
import time
import logging
import sqlite3
import datetime
import tkinter as tk
from tkinter import font as tkFont
from PIL import Image, ImageTk
import getpass

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Dlib face detector and models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')
face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")

class AttendanceSystem:
    def __init__(self):
        # Initialize Tkinter window
        self.win = tk.Tk()
        self.win.title("Smart Attendance System")
        self.win.geometry("1280x720")

        # Main container
        self.main_container = tk.Frame(self.win)
        self.main_container.pack(fill=tk.BOTH, expand=True)

        # Header Frame
        self.setup_header()

        # Camera Display Frame (Left)
        self.frame_left_camera = tk.Frame(self.main_container)
        self.camera_label = tk.Label(self.frame_left_camera)
        self.camera_label.pack(padx=10, pady=10)
        self.frame_left_camera.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Info Panel (Right)
        self.frame_right_info = tk.Frame(self.main_container)
        self.setup_info_panel()
        self.frame_right_info.pack(side=tk.RIGHT, padx=10, fill=tk.BOTH)

        # Initialize camera
        self.cap = self.get_camera_source()
        if not self.cap or not self.cap.isOpened():
            logger.error("No valid camera source found.")
            exit()

        # Initialize face recognition variables
        self.face_features_known_list = []
        self.face_name_known_list = []
        self.current_frame = None
        self.fps_show = 0
        self.start_time = time.time()

        # Setup database
        self.setup_database()
        # Load known faces
        self.load_known_faces()

        # Start clock update
        self.update_clock()

    def setup_header(self):
        header_frame = tk.Frame(self.win)
        header_frame.pack(fill=tk.X, padx=5, pady=5)

        # Current date and time
        self.clock_label = tk.Label(header_frame, 
                                  font=('Helvetica', 12))
        self.clock_label.pack(side=tk.LEFT)

        # Current user
        user_label = tk.Label(header_frame, 
                            text=f"User: {getpass.getuser()}", 
                            font=('Helvetica', 12))
        user_label.pack(side=tk.RIGHT)

    def update_clock(self):
        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.clock_label.config(text=f"Date/Time: {current_time}")
        self.win.after(1000, self.update_clock)

    def setup_info_panel(self):
        # Title
        self.font_title = tkFont.Font(family='Helvetica', size=20, weight='bold')
        tk.Label(self.frame_right_info, 
                text="Attendance System", 
                font=self.font_title).pack(pady=10)

        # Stats Frame
        stats_frame = tk.Frame(self.frame_right_info)
        stats_frame.pack(fill=tk.X, padx=5, pady=5)

        # FPS Display
        tk.Label(stats_frame, text="FPS: ").grid(row=0, column=0, sticky="w")
        self.label_fps = tk.Label(stats_frame, text="0")
        self.label_fps.grid(row=0, column=1, sticky="w")

        # Face Count Display
        tk.Label(stats_frame, text="Faces Detected: ").grid(row=1, column=0, sticky="w")
        self.label_face_count = tk.Label(stats_frame, text="0")
        self.label_face_count.grid(row=1, column=1, sticky="w")

        # Attendance Log
        log_frame = tk.Frame(self.frame_right_info)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        tk.Label(log_frame, text="Attendance Log:", 
                font=('Helvetica', 12, 'bold')).pack()

        # Create scrolled text widget for attendance log
        self.attendance_log = tk.Text(log_frame, height=15, width=40)
        scrollbar = tk.Scrollbar(log_frame, command=self.attendance_log.yview)
        self.attendance_log.configure(yscrollcommand=scrollbar.set)
        
        self.attendance_log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def get_camera_source(self):
        logger.info("Checking available camera sources...")

        # First try with default settings
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            try:
                # Try to set properties with error handling
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                
                actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                logger.info(f"Camera initialized: {actual_width}x{actual_height}")
                return cap
            except Exception as e:
                logger.warning(f"Could not set camera properties: {e}")
                return cap

        # Try alternative methods
        try:
            cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
            if cap.isOpened():
                logger.info("Using V4L2 backend")
                return cap
        except Exception as e:
            logger.warning(f"V4L2 attempt failed: {e}")

        try:
            cap = cv2.VideoCapture("/dev/video0")
            if cap.isOpened():
                logger.info("Using direct device access")
                return cap
        except Exception as e:
            logger.warning(f"Direct device access failed: {e}")

        logger.error("No working camera configuration found")
        return None

    def setup_database(self):
        try:
            conn = sqlite3.connect("attendance.db")
            cursor = conn.cursor()
            cursor.execute('''CREATE TABLE IF NOT EXISTS attendance
                            (name TEXT, time TEXT, date DATE, 
                            UNIQUE(name, date))''')
            conn.commit()
            conn.close()
            logger.info("Database setup complete")
        except Exception as e:
            logger.error(f"Database setup failed: {e}")

    def load_known_faces(self):
        try:
            if os.path.exists("data/features_all.csv"):
                csv_rd = pd.read_csv("data/features_all.csv", header=None)
                for i in range(csv_rd.shape[0]):
                    self.face_name_known_list.append(csv_rd.iloc[i][0])
                    features = [float(x) for x in csv_rd.iloc[i][1:129] if x != '']
                    self.face_features_known_list.append(features)
                logger.info(f"Loaded {len(self.face_features_known_list)} faces")
            else:
                logger.warning("features_all.csv not found!")
        except Exception as e:
            logger.error(f"Error loading faces: {e}")

    def mark_attendance(self, name):
        try:
            current_date = datetime.datetime.now().strftime('%Y-%m-%d')
            current_time = datetime.datetime.now().strftime('%H:%M:%S')
            
            conn = sqlite3.connect("attendance.db")
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM attendance WHERE name=? AND date=?", 
                         (name, current_date))
            
            if not cursor.fetchone():
                cursor.execute("INSERT INTO attendance VALUES (?, ?, ?)",
                             (name, current_time, current_date))
                conn.commit()
                log_message = f"[{current_time}] {name} marked present\n"
                self.attendance_log.insert(tk.END, log_message)
                self.attendance_log.see(tk.END)
            
            conn.close()
        except Exception as e:
            logger.error(f"Error marking attendance: {e}")

    def get_frame(self):
        if not self.cap or not self.cap.isOpened():
            logger.error("Camera is not opened!")
            return None, None

        try:
            ret, frame = self.cap.read()
            if not ret or frame is None or frame.size == 0:
                logger.warning("Failed to capture frame")
                return None, None

            if frame.shape[0] > 0 and frame.shape[1] > 0:
                if len(frame.shape) == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                
                frame = cv2.resize(frame, (640, 480))
                return ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                logger.warning("Invalid frame dimensions")
                return None, None

        except Exception as e:
            logger.error(f"Error capturing frame: {e}")
            return None, None

    def update_fps(self):
        now = time.time()
        self.fps_show = 1.0 / (now - self.start_time)
        self.start_time = now
        self.label_fps.configure(text=f"{self.fps_show:.2f}")

    def process_frame(self):
        try:
            ret, frame = self.get_frame()
            if frame is None:
                self.win.after(100, self.process_frame)
                return

            faces = detector(frame, 0)
            self.label_face_count.configure(text=str(len(faces)))

            for face in faces:
                try:
                    x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    shape = predictor(frame, face)
                    face_descriptor = face_reco_model.compute_face_descriptor(frame, shape)

                    if len(self.face_features_known_list) > 0:
                        distances = [np.linalg.norm(np.array(face_descriptor) - np.array(known_face))
                                   for known_face in self.face_features_known_list]
                        min_dist = min(distances)
                        if min_dist < 0.4:
                            name = self.face_name_known_list[distances.index(min_dist)]
                            cv2.putText(frame, name, (x1, y2 + 20), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                            self.mark_attendance(name)
                except Exception as e:
                    logger.error(f"Error processing face: {e}")
                    continue

            self.update_fps()
            try:
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                self.camera_label.imgtk = imgtk
                self.camera_label.configure(image=imgtk)
            except Exception as e:
                logger.error(f"Error updating display: {e}")

        except Exception as e:
            logger.error(f"Error in process_frame: {e}")
        finally:
            self.win.after(20, self.process_frame)

    def run(self):
        self.process_frame()
        self.win.mainloop()

    def __del__(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()

def main():
    attendance_system = AttendanceSystem()
    attendance_system.run()

if __name__ == "__main__":
    main()
