import cv2
from datetime import datetime
import os
import numpy as np
import tkinter as tk
from tkinter import ttk, Label
import threading
from queue import Queue, Empty
from PIL import Image, ImageTk

exit_program = False
flag = False
storage_dir = "events"
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("can't open camera")
    exit()

# Note 1. add red alarm bounding box.
# Note 2. Passing-by, Approaching, Suspicious detection.

class EventControlApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Surveillance System")
        self.root.geometry("800x600")

        self.events_state = [tk.BooleanVar(value=True) for _ in range(4)]

        self.setup_ui()

        self.event_queue = Queue()
        self.image_queue = Queue()

        self.thread = threading.Thread(target=video_processing,
                                       args=(cap, self.events_state, self.event_queue, self.image_queue))
        self.thread.start()

        self.update_image()
        self.root.after(100, self.check_queue)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def setup_ui(self):
        # set theme
        style = ttk.Style()
        style.theme_use('clam')

        # set framework
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left control panel
        control_frame = ttk.LabelFrame(main_frame, text="Event Controls", padding=(10, 5))
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        # Event control options
        self.create_event_controls(control_frame)

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

        event1_frame = ttk.Frame(parent)
        event1_frame.pack(fill=tk.X, pady=5)
        ttk.Label(event1_frame, text="Event 1: Person Detection Log").pack(side=tk.LEFT)
        self.event1_btn = ttk.Checkbutton(
            event1_frame,
            style="On.TCheckbutton",  # initial state as ON
            variable=self.events_state[0],
            command=lambda: self.update_button_style(0),
            text="ON" if self.events_state[0].get() else "OFF"  # ON/OFF
        )
        self.event1_btn.pack(side=tk.RIGHT)

        # Event 2 control
        event2_frame = ttk.Frame(parent)
        event2_frame.pack(fill=tk.X, pady=5)
        ttk.Label(event2_frame, text="Event 2: Capture Standard Photo").pack(side=tk.LEFT)
        self.event2_btn = ttk.Checkbutton(
            event2_frame,
            style="On.TCheckbutton",
            variable=self.events_state[1],
            command=lambda: self.update_button_style(1),
            text="ON" if self.events_state[1].get() else "OFF"
        )
        self.event2_btn.pack(side=tk.RIGHT)

        # Event 3 control
        event3_frame = ttk.Frame(parent)
        event3_frame.pack(fill=tk.X, pady=5)
        ttk.Label(event3_frame, text="Event 3: Capture HD Photo").pack(side=tk.LEFT)
        self.event3_btn = ttk.Checkbutton(
            event3_frame,
            style="On.TCheckbutton",
            variable=self.events_state[2],
            command=lambda: self.update_button_style(2),
            text="ON" if self.events_state[2].get() else "OFF"
        )
        self.event3_btn.pack(side=tk.RIGHT)

        # Event 4 control
        event4_frame = ttk.Frame(parent)
        event4_frame.pack(fill=tk.X, pady=5)
        ttk.Label(event4_frame, text="Event 4: Close-Range Recording").pack(side=tk.LEFT)
        self.event4_btn = ttk.Checkbutton(
            event4_frame,
            style="On.TCheckbutton",
            variable=self.events_state[3],
            command=lambda: self.update_button_style(3),
            text="ON" if self.events_state[3].get() else "OFF"
        )
        self.event4_btn.pack(side=tk.RIGHT)

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
                    n = task[0]
                    trigger(n, self.events_state)
                    self.status_bar.config(text=f"Event {n} triggered - {datetime.now().strftime('%H:%M:%S')}")
        except Empty:
            pass
        finally:
            self.root.after(100, self.check_queue)

    def on_close(self):
        global exit_program
        exit_program = True
        self.root.destroy()


def set_resolution(cap, width, height):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)


def log_event(event_type):
    with open("event_log.txt", "a") as log_file:
        log_file.write(f"{datetime.now()}: {event_type}\n")


def capture_photo(resolution, event_type, pic):
    ret, frame = pic.read()
    if ret:
        resized = cv2.resize(frame, resolution)
        filename = f"{storage_dir}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{event_type}.jpg"
        cv2.imwrite(filename, resized)


def exit_app(window):
    global exit_program
    exit_program = True
    window.destroy()


def trigger(n, events_state):
    global flag, cap
    if n == 1 and events_state[0].get():
        log_event("event1")
        print("event 1 write!")
    elif n == 2 and events_state[1].get():
        capture_photo((640, 480), "event2", cap)
        log_event("event2")
        print("event 2 write!")
    elif n == 3 and events_state[2].get():
        capture_photo((1920, 1080), "event3", cap)
        log_event("event3")
        print("event 3 write!")
    elif n == 4 and events_state[3].get():
        log_event("event4")
        flag = True
        print("event 4 write!")


def distance_finder(focal_length, real_face_width, face_width_in_frame):
    distance = (real_face_width * focal_length) / face_width_in_frame
    return distance


def check_queue(window,queue, events_state):
    try:
        while True:  # Process all available items in the queue
            task = queue.get_nowait()
            if task:
                n = task[0]
                trigger(n, events_state)
    except Empty:
        pass
    finally:
        window.after(1, check_queue, window, queue, events_state)  # Schedule next check


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


def video_processing(cap, events_state, queue, image_queue):
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
        # Read frame from camera
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to capture frame. Exiting...")
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
            for idx in indices:
                box = boxes[idx]
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
                        'confidence': confidences[idx],
                        'last_seen': frame_counter
                    })
                else:
                    # Add new object
                    tracked_objects[current_id] = {
                        'box': box,
                        'distance': distance,
                        'confidence': confidences[idx],
                        'last_seen': frame_counter
                    }
                    current_id += 1

                # Count people in different distance zones
                if distance < 300:
                    now[4] += 1
                elif distance < 500:
                    now[3] += 1
                elif distance < 1000:
                    now[2] += 1
                else:
                    now[1] += 1

            # Check for status changes and trigger events
            for j in range(1, 5):
                if now[j] != prestatus[j]:
                    queue.put((j,))
                    print(f"[EVENT] Triggering event {j}")
                if now[4] == prestatus[4]:
                    flag = False

            prestatus = now.copy()
            print(f"[DEBUG] Status array: {now}, Recording flag: {flag}")
            now.fill(0)

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

            ret, vidframe = cap.read()
            if ret:
                out2vid.write(vidframe)
        elif start_time != -1:  # If recording but event not active
            if (datetime.now() - start_time).seconds < duration:
                ret, vidframe = cap.read()
                if ret:
                    out2vid.write(vidframe)
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
    cap.release()
    print("[INFO] Video processing thread terminated")


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
    cap.release()


if __name__ == "__main__":
    main()
