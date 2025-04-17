import dlib
import numpy as np
import cv2
import os
import shutil
import time
import tkinter as tk
from tkinter import font as tkFont
from PIL import Image, ImageTk

# Initialize Dlib face detector
detector = dlib.get_frontal_face_detector()

class FaceRegister:
    def __init__(self):
        self.current_frame_faces_cnt = 0  # Number of faces in the current frame
        self.ss_cnt = 0  # Screenshot counter
        self.existing_faces_cnt = 0  # Number of faces in the database
        self.face_folder_created_flag = False  # Flag to check if face folder is created
        self.out_of_range_flag = False  # Flag to check if face is out of range

        # Face ROI coordinates
        self.face_ROI_height_start = 0
        self.face_ROI_height = 0
        self.face_ROI_width_start = 0
        self.face_ROI_width = 0

        # Tkinter GUI
        self.win = tk.Tk()
        self.win.title("Face Recognition System")
        self.win.geometry("1000x600")

        # Camera Display Frame
        self.frame_left_camera = tk.Frame(self.win)
        self.label = tk.Label(self.frame_left_camera)
        self.label.pack()
        self.frame_left_camera.pack(side=tk.LEFT)

        # Right Panel - Info and Controls
        self.frame_right_info = tk.Frame(self.win)
        self.label_fps_info = tk.Label(self.frame_right_info, text="FPS: ")
        self.label_face_cnt = tk.Label(self.frame_right_info, text="Faces in Frame: 0")
        self.label_warning = tk.Label(self.frame_right_info, text="", fg="red")
        self.log_all = tk.Label(self.frame_right_info, text="")
        self.input_name = tk.Entry(self.frame_right_info)
        self.input_name_char = ""

        self.font_title = tkFont.Font(family='Helvetica', size=20, weight='bold')
        self.font_step_title = tkFont.Font(family='Helvetica', size=15, weight='bold')

        # Organizing UI Layout
        tk.Label(self.frame_right_info, text="Face Recognition System", font=self.font_title).grid(row=0, column=0, columnspan=3, pady=10)

        tk.Label(self.frame_right_info, text="FPS: ").grid(row=1, column=0, padx=5, pady=2, sticky="w")
        self.label_fps_info.grid(row=1, column=1, padx=5, pady=2, sticky="w")

        tk.Label(self.frame_right_info, text="Faces in Frame: ").grid(row=2, column=0, padx=5, pady=2, sticky="w")
        self.label_face_cnt.grid(row=2, column=1, padx=5, pady=2, sticky="w")

        self.label_warning.grid(row=3, column=0, columnspan=2, pady=5)

        # Step 1: Clear old data
        tk.Label(self.frame_right_info, text="Step 1: Clear face photos", font=self.font_step_title).grid(row=4, column=0, columnspan=2, pady=10)
        tk.Button(self.frame_right_info, text="Clear", command=self.GUI_clear_data).grid(row=5, column=0, columnspan=2, pady=5)

        # Step 2: Input name
        tk.Label(self.frame_right_info, text="Step 2: Input name", font=self.font_step_title).grid(row=6, column=0, columnspan=2, pady=10)
        tk.Label(self.frame_right_info, text="Name: ").grid(row=7, column=0, padx=5, pady=2, sticky="w")
        self.input_name.grid(row=7, column=1, padx=5, pady=2, sticky="w")
        tk.Button(self.frame_right_info, text="Input", command=self.GUI_get_input_name).grid(row=7, column=2, padx=5, pady=2)

        # Step 3: Save current face
        tk.Label(self.frame_right_info, text="Step 3: Save face image", font=self.font_step_title).grid(row=8, column=0, columnspan=2, pady=10)
        tk.Button(self.frame_right_info, text="Save current face", command=self.save_current_face).grid(row=9, column=0, columnspan=2, pady=5)

        self.log_all.grid(row=10, column=0, columnspan=2, pady=10)

        self.frame_right_info.pack(side=tk.RIGHT, padx=10)

        self.fps_show = 0
        self.start_time = time.time()

        # Try to find a working camera
        self.cap = self.get_camera_source()

        if not self.cap or not self.cap.isOpened():
            print("❌ Error: No valid camera source found.")
            exit()

        # Path for saving face images
        self.path_photos_from_camera = "data/data_faces_from_camera/"
        self.pre_work_mkdir()

    def get_camera_source(self):
        print("🔍 Checking available camera sources...")

        cap = cv2.VideoCapture(0, cv2.CAP_V4L2)  # Force V4L2
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            print("✅ Using OpenCV VideoCapture(0) with 640x480 resolution")
            return cap

        cap = cv2.VideoCapture("/dev/video0")
        if cap.isOpened():
            print("✅ Using /dev/video0 (V4L2)")
            return cap

        print("❌ No working camera detected.")
        return None

    def get_frame(self):
        if not self.cap or not self.cap.isOpened():
            print("❌ Error: Camera is not opened! Retrying...")
            self.cap.release()
            time.sleep(2)
            self.cap = self.get_camera_source()
            return None, None  

        ret, frame = self.cap.read()

        if not ret or frame is None or frame.size == 0:
            print("❌ Warning: No frame captured. Retrying...")
            return None, None  

        print(f"✅ Frame captured! Shape: {frame.shape}")

        if frame.shape[0] == 1 and frame.shape[1] > 1000:
            print("❌ Error: Frame is not correctly shaped. Reshaping...")
            frame = frame.reshape((480, 640, 3))

        return ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def process(self):
        ret, self.current_frame = self.get_frame()
        if self.current_frame is None:
            self.win.after(20, self.process)
            return

        faces = detector(self.current_frame, 0)

        # Update the number of faces in the current frame
        self.current_frame_faces_cnt = len(faces)
        self.label_face_cnt["text"] = f"Faces in Frame: {self.current_frame_faces_cnt}"

        # Draw rectangle around faces and update ROI coordinates
        for d in faces:
            # Increase the bounding box size by 20% to capture more of the face
            padding = 0.2  # 20% padding around the face
            x1 = max(0, int(d.left() - padding * (d.right() - d.left())))
            y1 = max(0, int(d.top() - padding * (d.bottom() - d.top())))
            x2 = min(self.current_frame.shape[1], int(d.right() + padding * (d.right() - d.left())))
            y2 = min(self.current_frame.shape[0], int(d.bottom() + padding * (d.bottom() - d.top())))

            self.current_frame = cv2.rectangle(self.current_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Update face ROI coordinates
            self.face_ROI_height_start = y1
            self.face_ROI_height = y2 - y1
            self.face_ROI_width_start = x1
            self.face_ROI_width = x2 - x1

        try:
            img_Image = Image.fromarray(self.current_frame)
            img_PhotoImage = ImageTk.PhotoImage(image=img_Image)
            self.label.img_tk = img_PhotoImage
            self.label.configure(image=img_PhotoImage)
        except Exception as e:
            print("❌ Error converting frame to Tkinter format:", e)

        self.win.after(20, self.process)

    def update_fps(self):
        now = time.time()
        self.fps_show = 1.0 / (now - self.start_time)
        self.start_time = now
        self.label_fps_info["text"] = f"FPS: {self.fps_show:.2f}"

    def pre_work_mkdir(self):
        if not os.path.isdir(self.path_photos_from_camera):
            os.mkdir(self.path_photos_from_camera)

    def GUI_clear_data(self):
        folders_rd = os.listdir(self.path_photos_from_camera)
        for i in range(len(folders_rd)):
            shutil.rmtree(self.path_photos_from_camera + folders_rd[i])
        self.label_face_cnt["text"] = "0"
        self.existing_faces_cnt = 0
        self.log_all["text"] = "Face images removed!"

    def GUI_get_input_name(self):
        self.input_name_char = self.input_name.get()
        self.create_face_folder()
        self.log_all["text"] = f"Name '{self.input_name_char}' saved!"

    def create_face_folder(self):
        self.existing_faces_cnt += 1
        if self.input_name_char:
            self.current_face_dir = self.path_photos_from_camera + "person_" + str(self.existing_faces_cnt) + "_" + self.input_name_char
        else:
            self.current_face_dir = self.path_photos_from_camera + "person_" + str(self.existing_faces_cnt)
        os.makedirs(self.current_face_dir, exist_ok=True)  # Ensure the folder exists
        self.face_folder_created_flag = True
        self.log_all["text"] = f"Folder '{self.current_face_dir}' created!"

    def save_current_face(self):
        if self.face_folder_created_flag:
            if self.current_frame_faces_cnt == 1:
                if not self.out_of_range_flag:
                    self.ss_cnt += 1
                    # Ensure the ROI coordinates are within the frame dimensions
                    if (self.face_ROI_height_start >= 0 and self.face_ROI_width_start >= 0 and
                        self.face_ROI_height_start + self.face_ROI_height <= self.current_frame.shape[0] and
                        self.face_ROI_width_start + self.face_ROI_width <= self.current_frame.shape[1]):
                        face_image = self.current_frame[
                            self.face_ROI_height_start:self.face_ROI_height_start + self.face_ROI_height,
                            self.face_ROI_width_start:self.face_ROI_width_start + self.face_ROI_width
                        ]
                        cv2.imwrite(f"{self.current_face_dir}/img_face_{self.ss_cnt}.jpg", cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR))
                        self.log_all["text"] = f"Face saved to '{self.current_face_dir}/img_face_{self.ss_cnt}.jpg'"
                    else:
                        self.log_all["text"] = "Face is out of range!"
                else:
                    self.log_all["text"] = "Face is out of range!"
            else:
                self.log_all["text"] = "No face detected in the current frame!"
        else:
            self.log_all["text"] = "Please input a name first!"

    def run(self):
        self.process()
        self.win.mainloop()

def main():
    FaceRegister().run()

if __name__ == '__main__':
    main()
