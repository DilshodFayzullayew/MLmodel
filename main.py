import tkinter as tk
from tkinter import messagebox
import os
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import sys
import os
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
def run_yolo_video_processing():
    frame_counter = 0
    try:
        VIDEO_DIR = os.path.join('.', 'videos')
        video_path = os.path.join(VIDEO_DIR, 'v5.mp4')
        OUTPUT_DIR = os.path.join('.', 'output_crops')
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        # desired_width = 840
        # desired_height = 800
        # Load the YOLO model
        model_path = "train/weights/best.pt"
        model = YOLO(model_path)

        # Set the detection threshold
        threshold = 0.7  # Set threshold to 70%
        class_name_dict = {0: 'elektrovoz',
                           1: 'elektrovoz_number',
                           2: 'teplovoz',
                           3: 'teplovoz_number'}

        # Classes to crop
        classes_to_crop = [1, 3]  # 1: elektrovoz_number, 3: teplovoz_number

        # Process the video frame by frame
        while ret:
            # frame = cv2.resize(frame, (desired_width, desired_height))

            results = model(frame)[0]
            for result in results.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = result
                if score > 0.2 and int(class_id) in classes_to_crop:
                    # Draw bounding box and label on the resized frame
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                    cv2.putText(frame, class_name_dict.get(int(class_id), 'UNKNOWN').upper(),
                                (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
                    if score > 0.6:
                        cropped_img = frame[int(y1):int(y2), int(x1):int(x2)]
                        crop_filename = os.path.join(OUTPUT_DIR, f'frame_{frame_counter}_crop_{int(x1)}_{int(y1)}.jpg')
                        cv2.imwrite(crop_filename, cropped_img)

            # Display the resized frame (optional)
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Read the next frame
            ret, frame = cap.read()
            frame_counter += 1

        # Release resources
        cap.release()
        cv2.destroyAllWindows()

    except Exception as e:
        messagebox.showerror("Error", str(e))

# Create the main Tkinter window
root = tk.Tk()
root.title("YOLO Video Processing")
root.geometry("800x600")  # Set the size of the window (width x height)

# Create a label widget
label = tk.Label(root, text="Click the button to process the video using YOLO.")
label.pack(pady=20)

# Create a button to run the YOLO video processing function
button = tk.Button(root, text="Run YOLO", command=run_yolo_video_processing)
button.pack(pady=10)

# Run the Tkinter event loop
root.mainloop()

sys.stderr = stderr