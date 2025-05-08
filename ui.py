import cv2
from datetime import datetime
import os
import time
import numpy as np
import tkinter as tk
from tkinter import ttk, Label
import threading
from queue import Queue, Empty
from PIL import Image, ImageTk

import builtins
import tkinter.scrolledtext as scrolledtext
from pir_sensor import PIRSensorController


exit_program = False
flag = False
storage_dir = "events"
caps = [cv2.VideoCapture(0), cv2.VideoCapture(1)]
for i, cam in enumerate(caps):
    if not cam.isOpened():
        print(f"can't open camera {i}")
        exit()

# Note 1. add red alarm bounding box.
# Note 2. Passing-by, Approaching, Suspicious detection.
# Thread-safe lock and auto-switch flag
mode_lock = threading.Lock()
auto_switch = False
detection_flag = False

pir_states = {0: False, 1: False}
pir_controller = None


class EventControlApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Surveillance System")
        self.root.geometry("800x600")

        self.events_state = [tk.BooleanVar(value=True) for _ in range(4)]
        self.cam_mode = tk.StringVar(value="manual")
        self.pir_mode = tk.BooleanVar(value=False)
        self.selected_camera = tk.IntVar(value=0)

        self._orig_print = builtins.print
        builtins.print = self._gui_print

        self.setup_ui()

        self.event_queue = Queue()
        self.image_queue = Queue()
        self.log_queue = Queue()

        self.thread = threading.Thread(
            target=video_processing,
            args=(caps, self.selected_camera, self.events_state, self.event_queue, self.image_queue),
            daemon=True
        )
        self.thread.start()

        self.update_image()
        self.root.after(100, self.check_queue)
        self.root.after(100, self._flush_logs)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def setup_ui(self):
        # set theme
        style = ttk.Style()
        style.theme_use('clam')

        # set framework
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left control panel
        control_frame = ttk.LabelFrame(main_frame, text="Mode Controls", padding=(10, 5))
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        # Event control options
        self.create_event_controls(control_frame)
        self.create_camera_controls(control_frame)

        # log frame
        # log_frame = ttk.LabelFrame(self.root, text="Log")
        # log_frame.place(relx=0.025, rely=0.65, relwidth=0.32, relheight=0.23)
        log_frame = ttk.LabelFrame(self.root, text="Log")
        log_frame.place(x=20, y=390, width=256, height=138)

        # ScrolledText scrolling control
        self.log_text = scrolledtext.ScrolledText(log_frame, state='disabled', wrap='word', height=8)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # Video display area
        video_frame = ttk.LabelFrame(main_frame, text="Live Monitoring", padding=(5, 5))
        video_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.video_label = Label(video_frame, bg='black')
        self.video_label.pack(fill=tk.BOTH, expand=True)

        # Status bar
        self.status_bar = ttk.Label(main_frame, text="System Ready", relief=tk.SUNKEN)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X, pady=(5, 0))


        # Exit button
        exit_btn = ttk.Button(control_frame, text="Exit System", command=self.on_close)
        exit_btn.pack(side=tk.BOTTOM, pady=(10, 0), fill=tk.X)

    def update_button_style(self, event_index):
        buttons = [self.event1_btn, self.event2_btn, self.event3_btn, self.event4_btn]
        btn = buttons[event_index]

        if self.events_state[event_index].get():
            btn.configure(style="On.TCheckbutton", text="ON")
        else:
            btn.configure(style="Off.TCheckbutton", text="OFF")

    def create_event_controls(self, parent):
        # Event 1 control
        style = ttk.Style()
        style.configure("On.TCheckbutton", foreground="green", font=('Helvetica', 10, 'bold'))
        style.configure("Off.TCheckbutton", foreground="red", font=('Helvetica', 10))

        events = ["Event 1: Person Detection Log", "Event 2: Capture Standard Photo",
                  "Event 3: Capture HD Photo", "Event 4: Close-Range Recording"]
        for i, text in enumerate(events):
            frame = ttk.Frame(parent)
            frame.pack(fill=tk.X, pady=5)
            ttk.Label(frame, text=text).pack(side=tk.LEFT)
            btn = ttk.Checkbutton(
                frame, style="On.TCheckbutton", variable=self.events_state[i],
                command=lambda idx=i: self.update_button_style(idx),
                text="ON" if self.events_state[i].get() else "OFF"
            )
            setattr(self, f"event{i + 1}_btn", btn)
            btn.pack(side=tk.RIGHT)

    def update_button_style(self, idx):
        btn = getattr(self, f"event{idx+1}_btn")
        if self.events_state[idx].get(): btn.configure(style="On.TCheckbutton", text="ON")
        else: btn.configure(style="Off.TCheckbutton", text="OFF")

    def create_camera_controls(self, parent):
        cam_frame = ttk.LabelFrame(parent, text="Camera Controls", padding=(10,5))
        cam_frame.pack(fill=tk.X, pady=10)
        ttk.Radiobutton(cam_frame, text="Manual", variable=self.cam_mode, value="manual", command=self.on_cam_mode_change).pack(anchor=tk.W)
        ttk.Radiobutton(cam_frame, text="Auto (5s)", variable=self.cam_mode, value="auto", command=self.on_cam_mode_change).pack(anchor=tk.W)
        man = ttk.Frame(cam_frame); man.pack(fill=tk.X, pady=5)
        ttk.Button(man, text="Camera 1", command=lambda: self.switch_camera(0)).pack(side=tk.LEFT, expand=True)
        ttk.Button(man, text="Camera 2", command=lambda: self.switch_camera(1)).pack(side=tk.RIGHT, expand=True)
        self.cam_label = ttk.Label(cam_frame, text="Current Camera: 1"); self.cam_label.pack(pady=5)

        # PIR Polling
        pir_frame = ttk.Frame(parent)
        pir_frame.pack(fill=tk.X, pady=5)
        ttk.Checkbutton(
            pir_frame,
            text="PIR Polling Mode",
            variable=self.pir_mode,
            command=self.on_pir_mode_toggle
        ).pack(side=tk.LEFT)

    def on_pir_mode_toggle(self):
        global pir_controller, auto_switch

        if self.pir_mode.get():
            # turn on PIR mode，停止自动切换
            auto_switch = False
            if pir_controller is None:
                pir_controller = PIRSensorController(pir_pins=[3, 4], states=pir_states)
                pir_controller.start()
            self.status_bar.config(text="PIR Mode ON")
        else:
            # turn off PIR mode
            if pir_controller:
                pir_controller.stop()
            pir_controller = None
            self.status_bar.config(text="PIR Mode OFF")

    def on_cam_mode_change(self):
        global auto_switch
        if self.cam_mode.get() == "auto":
            auto_switch = True
            self.schedule_auto_switch()
        else:
            auto_switch = False
            self.root.after_cancel(self.auto_job) if hasattr(self,'auto_job') else None

    def schedule_auto_switch(self):
        self.auto_job = self.root.after(5000, self.auto_switch_camera)

    def auto_switch_camera(self):
        global detection_flag
        if not detection_flag:
            nxt = 1 - self.selected_camera.get()
            self.switch_camera(nxt)
        self.schedule_auto_switch()

    def switch_camera(self, idx):
        with mode_lock:
            self.selected_camera.set(idx)
        self.cam_label.config(text=f"Current Camera: {idx+1}")

    def update_image(self):
        try:
            pil_image = self.image_queue.get_nowait()
            # Maintain aspect ratio while resizing
            img_width, img_height = pil_image.size
            ratio = min(600 / img_width, 450 / img_height)
            new_size = (int(img_width * ratio), int(img_height * ratio))
            pil_image = pil_image.resize(new_size, Image.LANCZOS)

            tk_image = ImageTk.PhotoImage(image=pil_image)
            self.video_label.config(image=tk_image)
            self.video_label.image = tk_image
        except Empty:
            pass
        finally:
            self.root.after(10, self.update_image)

    def check_queue(self):
        try:
            while True:
                task = self.event_queue.get_nowait()
                if task:
                    # 解包出事件编号和摄像头索引
                    n, cam_idx = task
                    # 传入 cam_idx 给 trigger
                    trigger(n, self.events_state, cam_idx)
                    # 在状态栏显示是哪个摄像头触发的
                    self.status_bar.config(
                        text=f"Event {n} triggered on camera {cam_idx+1} — {datetime.now().strftime('%H:%M:%S')}"
                    )
        except Empty:
            pass
        finally:
            self.root.after(100, self.check_queue)









    def on_close(self):
        global exit_program
        exit_program = True
        self.root.destroy()

    def _gui_print(self, *args, **kwargs):
        msg = " ".join(str(a) for a in args)
        self.log_queue.put(msg)
        self._orig_print(msg, **kwargs)

    def _flush_logs(self):
        try:
            while True:
                msg = self.log_queue.get_nowait()
                self.log_text.config(state='normal')
                self.log_text.insert('end', msg + '\n')
                self.log_text.see('end')
                self.log_text.config(state='disabled')
        except Empty:
            pass
        finally:
            self.root.after(100, self._flush_logs)


def set_resolution(cap, width, height):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)


def log_event(event_type):
    with open("event_log.txt", "a") as log_file:
        log_file.write(f"{datetime.now()}: {event_type}\n")


def capture_photo(resolution, event_type, cam):
    ret, frame = cam.read()
    if ret:
        resized = cv2.resize(frame, resolution)
        filename = f"{storage_dir}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{event_type}.jpg"
        cv2.imwrite(filename, resized)


def exit_app(window):
    global exit_program
    exit_program = True
    window.destroy()


def trigger(n, events_state, cam_idx):
    global flag, cap
    if n == 1 and events_state[0].get():
        log_event("event1")
        print("event 1 write!")
    elif n == 2 and events_state[1].get():
        capture_photo((640, 480), "event2", caps[cam_idx])
        log_event("event2")
        print("event 2 write!")
    elif n == 3 and events_state[2].get():
        capture_photo((1920, 1080), "event3", caps[cam_idx])
        log_event("event3")
        print("event 3 write!")
    elif n == 4 and events_state[3].get():
        log_event("event4")
        flag = True
        print("event 4 write!")


def distance_finder(focal_length, real_face_width, face_width_in_frame):
    distance = (real_face_width * focal_length) / face_width_in_frame
    return distance


# def check_queue(window, queue, events_state):
#     try:
#         while True:  # Process all available items in the queue
#             task = queue.get_nowait()
#             if task:
#                 n = task[0]
#                 trigger(n, events_state)
#     except Empty:
#         pass
#     finally:
#         window.after(1, check_queue, window, queue, events_state)  # Schedule next check


def focal_length(measured_distance, real_width, width_in_rf_image):
    focal_length_value = (width_in_rf_image * measured_distance) / real_width
    return focal_length_value


def update_image(label, image_queue):
    try:
        pil_image = image_queue.get_nowait()
        tk_image = ImageTk.PhotoImage(image=pil_image)
        label.config(image=tk_image)
        label.image = tk_image
    except Empty:
        pass
    finally:
        label.after(1, update_image, label, image_queue)


def video_processing(caps, cam_index_var, events_state, queue, image_queue):
    """
    Main video processing function that runs in a separate thread.
    Handles object detection, distance calculation, event triggering and video recording.
    """
    # Initialize status arrays for tracking people in different distance zones
    prestatus = np.zeros(5, dtype=int)  # Previous frame's status array
    now = np.zeros(5, dtype=int)  # Current frame's status array

    # Create storage directory if it doesn't exist
    os.makedirs(storage_dir, exist_ok=True)

    # Video recording parameters
    duration = 2  # Recording duration in seconds
    global flag  # Event 4 (recording) flag
    global exit_program  # Program exit flag
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Video codec

    # Object tracking variables
    tracked_objects = {}  # Dictionary to track detected objects
    current_id = 0  # Counter for assigning unique IDs
    display_duration = 30  # Number of frames to keep displaying an object

    # Distance measurement calibration parameters
    start_time = -1  # Recording start time
    real_distance = 156.5  # Calibration distance (cm)
    real_width = 45  # Calibration object width (cm)
    width_in_frame = 359  # Calibration object width in pixels
    human_wide = 45  # Assumed average human width (cm)
    frame_counter = 0  # Frame counter for detection frequency

    # Load YOLO model
    modelConfiguration = "yolov4-tiny.cfg"
    modelWeights = "yolov4-tiny.weights"
    net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # Calculate focal length using calibration parameters
    focal = focal_length(real_distance, real_width, width_in_frame)

    # Main processing loop
    while not exit_program:
        # Select active camera
        with mode_lock:
            idx = cam_index_var.get()
        current_idx = idx

        if pir_controller is not None and not pir_states[idx]:
            # 动态获取摄像头分辨率
            width = int(caps[idx].get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(caps[idx].get(cv2.CAP_PROP_FRAME_HEIGHT))
            blank = np.zeros((height, width, 3), dtype=np.uint8)
            pil_image = Image.fromarray(blank)
            image_queue.put(pil_image)
            time.sleep(0.05)  # 减少 CPU 占用
            continue

        ret, frame = caps[idx].read()
        if not ret:
            print(f"[ERROR] Failed to capture from camera {idx}.")
            break

        frame_counter += 1

        # Clean up old tracked objects
        to_delete = [obj_id for obj_id, obj in tracked_objects.items()
                     if frame_counter - obj['last_seen'] > display_duration]
        for obj_id in to_delete:
            del tracked_objects[obj_id]

        # Perform detection every 20 frames (balance between performance and accuracy)
        if frame_counter % 20 == 0:
            boxes = []
            confidences = []

            # Prepare image for YOLO network
            blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
            net.setInput(blob)
            outs = net.forward(net.getUnconnectedOutLayersNames())

            frame_height, frame_width = frame.shape[:2]
            # print(frame_height, frame_width)

            # Process detection results
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]

                    # Only process 'person' class (class_id=0) with confidence > 50%
                    if (class_id == 0) and (confidence > 0.5):
                        # Convert bounding box coordinates from relative to absolute
                        center_x = int(detection[0] * frame_width)
                        center_y = int(detection[1] * frame_height)
                        w = int(detection[2] * frame_width)
                        h = int(detection[3] * frame_height)
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))

            # Apply Non-Maximum Suppression to eliminate redundant overlapping boxes
            indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

            # Update tracked objects with new detections
            # for idx in indices:
            #     box = boxes[idx]
            # Apply Non-Maximum Suppression ...
            for det_idx in indices:
                box = boxes[det_idx]
                x, y, w, h = box[0], box[1], box[2], box[3]
                distance = distance_finder(focal, human_wide, w)

                # Try to match with existing tracked objects
                matched_id = None
                for obj_id, obj in tracked_objects.items():
                    iou = calculate_iou(box, obj['box'])
                    if iou > 0.3:  # If overlap > 30%, consider it the same object
                        matched_id = obj_id
                        break

                if matched_id is not None:
                    # Update existing object
                    tracked_objects[matched_id].update({
                        'box': box,
                        'distance': distance,
                        'confidence': confidences[det_idx],
                        'last_seen': frame_counter
                    })
                else:
                    # Add new object
                    tracked_objects[current_id] = {
                        'box': box,
                        'distance': distance,
                        'confidence': confidences[det_idx],
                        'last_seen': frame_counter
                    }
                    current_id += 1

                # Count people in different distance zones
                if distance < 100:
                    now[4] += 1
                elif distance < 300:
                    now[3] += 1
                elif distance < 500:
                    now[2] += 1
                now[1] += 1

            # Check for status changes and trigger events
            for j in range(1, 5):
                if now[j] != prestatus[j]:
                    queue.put((j, idx))
                    print(f"[EVENT] Triggering event {j}")
                if now[4] == prestatus[4]:
                    flag = False

            prestatus = now.copy()
            # print(f"[DEBUG] Status array: {now}, Recording flag: {flag}")
            now.fill(0)

            # -----------------------------
            global detection_flag
            detection_flag = prestatus[1:].sum() > 0
            # -----------------------------

        # Draw all tracked objects (including recently lost ones)
        for obj_id, obj in tracked_objects.items():
            x, y, w, h = obj['box']

            # Calculate transparency based on how recently the object was seen
            alpha = min(1.0, 1.0 - (frame_counter - obj['last_seen']) / display_duration * 0.7)

            # Draw bounding box with transparency
            overlay = frame.copy()
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

            # Display information with same transparency
            distance_label = f"Distance: {obj['distance']:.2f} cm"
            label = f"Person: {obj['confidence'] * 100:.2f}%"

            text_overlay = frame.copy()
            cv2.putText(text_overlay, distance_label, (x, y - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(text_overlay, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            frame = cv2.addWeighted(text_overlay, alpha, frame, 1 - alpha, 0)

        # Handle recording (Event 4)
        if flag:  # If recording is triggered
            if start_time == -1:  # If not currently recording
                filename = f"{storage_dir}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_4.avi"
                out2vid = cv2.VideoWriter(filename, fourcc, 20.0, (640, 480))
                start_time = datetime.now()
                print(f"[RECORDING] Started recording: {filename}")
            else:
                start_time = datetime.now()  # Keep updating start time

            ret2, f2 = caps[idx].read()
            if ret2 and out2vid: out2vid.write(f2)
        elif start_time != -1:  # If recording but event not active
            if (datetime.now() - start_time).seconds < duration:
                ret2, f2 = caps[idx].read()
                if ret2: out2vid.write(f2)
            else:
                flag = False
                start_time = -1
                out2vid.release()
                print("[RECORDING] Stopped recording")

        # Convert frame to RGB and put in queue for display
        color_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(color_image)
        image_queue.put(pil_image)

    # Release camera when done
    for cam in caps: cam.release()
    print("[INFO] Video thread ended")


def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes
    Args:
        box1: [x1, y1, w1, h1]
        box2: [x2, y2, w2, h2]
    Returns:
        iou: Intersection over Union ratio (0.0 to 1.0)
    """
    # Determine coordinates of intersection area
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[0] + box1[2], box2[0] + box2[2])
    y_bottom = min(box1[1] + box1[3], box2[1] + box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # Calculate intersection area
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate areas of each box
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]

    # Calculate union area
    union_area = box1_area + box2_area - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area
    return iou


def main():
    root = tk.Tk()
    app = EventControlApp(root)
    root.mainloop()

    app.thread.join()


if __name__ == "__main__":
    main()
