# Video Surveillance System

This project implements a local, event-driven video surveillance system using a Raspberry Pi 5, dual USB cameras, and PIR sensors. Key features include real-time person detection with Tiny-YOLO, area-based distance estimation, configurable event triggers, and a Tkinter GUI for manual control and status monitoring. All processing and storage are performed locally without cloud dependencies.

---

## Features

* **Multi-Sensor Fusion**: Combines vision-based detection (Tiny-YOLO via OpenCV) and PIR motion sensing for reliable activation.
* **Distance Estimation**: Uses an area-based pinhole camera formula to bucket detected persons into four zones (critical, close, mid, far).
* **Event Triggers**: Four distinct actions based on distance: idle logging, standard photo capture (640×480), HD photo capture (1920×1080), and short video recording (2 seconds).
* **PIR Polling Mode**: Optionally gate camera activation by PIR sensors to conserve CPU and power.
* **Queue-Based Architecture**: Decouple real-time detection from GUI updates and storage I/O to maintain responsiveness.
* **Storage Management**: Timestamped JPEG and AVI files saved to `events/` directory. Future circular-buffer cleanup and external USB storage support.
* **Tkinter GUI**: Manual/Auto/PIR control, event toggles, live video preview, and embedded log console.

---

## Repository Structure

```
├── ui.py                  # Main application with GUI and video-processing thread
├── pir_sensor.py          # PIR sensor polling controller module
├── events/                # Directory for captured images and video clips
├── event_log.txt          # Persistent log of triggered events
├── yolov4-tiny.cfg        # Tiny-YOLO configuration file
├── yolov4-tiny.weights    # Tiny-YOLO weights file
└── README.md              # Project overview and setup instructions
```

---

## Dependencies

* Python 3.8+
* OpenCV (`opencv-python`)
* NumPy
* Pillow (`PIL`)
* Tkinter (usually included with Python)
* RPi.GPIO (on Raspberry Pi) or `fake-rpi` for macOS development

Install Python dependencies via pip:

```bash
pip install opencv-python numpy Pillow fake-rpi
```

On Raspberry Pi, install real GPIO library:

```bash
pip install RPi.GPIO
```

---

## Setup and Usage

1. **Clone the repository**:

   ```bash
   git clone git@github.com:ChengwZhou/ECSE488-Embedded-System-Design.git
   ```

git clone <repository-url>
cd <repository-folder>

````

2. **Prepare hardware**:
   - Connect two USB cameras to the Pi’s USB 3.0 ports.  
   - Wire PIR sensors to GPIO pins 3 and 4 and ground.  

3. **Download YOLO files**:
   Ensure `yolov4-tiny.cfg` and `yolov4-tiny.weights` are in the project root.

4. **Run the application**:
   ```bash
python ui.py
````

5. **GUI Controls**:

   * Select Manual, Auto (5-second switch), or PIR Polling mode.
   * Use camera buttons to manually switch streams.
   * Toggle each event type ON/OFF.
   * Monitor live video, logs, and status bar.

6. **Shutdown**:
   Click the "Exit System" button or close the window to ensure graceful cleanup of cameras and GPIO.

---

## Calibration and Configuration

* **Distance Estimation**: Modify calibration parameters (`ref_length`, `real_bb_area`) in `ui.py` based on a reference object at a known distance.
* **Polling Interval**: Adjust PIR polling frequency (`poll_interval`) in `pir_sensor.py` for responsiveness vs. CPU usage.
* **Frame Rate**: Change detection frequency by modifying the frame skip interval in `video_processing()`.

---

[//]: # (## Future Enhancements)

[//]: # ()
[//]: # (* Circular-buffer file management to prevent SD card fill-up.)

[//]: # (* External USB storage auto-migration.)

[//]: # (* Alternative lightweight detection model for higher frame rates.)

[//]: # (* Remote web UI and notifications &#40;email/SMS&#41; for alerts.)

[//]: # ()
[//]: # (---)

**Author:** [Lauren Hong](https://github.com/lhong03), [Chengwei Zhou](https://github.com/ChengwZhou)
---
**Date:** May 2025
