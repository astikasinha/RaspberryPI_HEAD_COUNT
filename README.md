

import cv2
import serial
import time
import gc  # For garbage collection
import os
import psutil

# Initialize serial port
ser = serial.Serial(
    port='/dev/ttyS0',
    baudrate=9600,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
    bytesize=serial.EIGHTBITS,
    timeout=1
)
ser.reset_input_buffer()
ser.reset_output_buffer()

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Open camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 500)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 390)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
mid_line = frame_height // 3

# Face tracking
trackers = {}  # Format: id -> (x, y, side, last_seen_time)
next_id = 1
crossing_count = 0

# Helper: Determine which side of line
def get_line_side(y, line_position):
    return "above" if y < line_position else "below"

# Helper: Log memory usage
def log_memory():
    process = psutil.Process(os.getpid())
    print(f"Memory used: {process.memory_info().rss / 1024**2:.2f} MB")

# Main loop
while cap.isOpened():
    for _ in range(5):  
        cap.grab()
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
    cv2.line(frame, (0, mid_line), (frame_width, mid_line), (0, 0, 255), 2)

    updated_trackers = {}
    current_time = time.time()

    for (x, y, w, h) in faces:
        center_x = x + w // 2
        center_y = y + h // 2

        matched_id = None
        for track_id, (prev_x, prev_y, prev_side, last_seen) in trackers.items():
            distance = ((center_x - prev_x) ** 2 + (center_y - prev_y) ** 2) ** 0.5
            if distance < 50:
                matched_id = track_id
                break

        if matched_id is None:
            matched_id = next_id
            next_id += 1

        current_side = get_line_side(center_y, mid_line)
        prev_side = trackers.get(matched_id, (0, 0, "none", 0))[2]

        # If crossing occurred
        if prev_side != current_side:
            if (prev_side == "above" and current_side == "below") or (prev_side == "below" and current_side == "above"):
                crossing_count += 1
                print(f"Face ID {matched_id} crossed line: {prev_side} -> {current_side}. Count = {crossing_count}")

        updated_trackers[matched_id] = (center_x, center_y, current_side, current_time)

        # Draw face and ID
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, f"ID {matched_id}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Remove old/stale trackers (not seen for 2 seconds)
    trackers = {tid: data for tid, data in updated_trackers.items() if current_time - data[3] < 2}

    # Show count on frame
    cv2.putText(frame, f"Count: {crossing_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show video
    cv2.imshow('Webcam Face Detection', frame)

    # Send count via serial
    try:
        ser.write(f"counter:{crossing_count}\n".encode())
    except Exception as e:
        print(f"Serial write error: {e}")

    # Periodically clean memory
    if next_id % 10 == 0:
        gc.collect()
        log_memory()

    # Exit on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Final cleanup
cap.release()
cv2.destroyAllWindows()
ser.write(f"counter:{crossing_count}\n".encode())
ser.reset_input_buffer()
ser.reset_output_buffer()
print("Processing complete.")
