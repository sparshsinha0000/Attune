import os
import sys
import time
import cv2
from ultralytics import YOLO

# =========================================================
# AI OBJECT DETECTOR - PORTABLE WINDOWS PROTOTYPE
# =========================================================
# The program uses a pretrained YOLO26 nano detector.
# On the first run, Ultralytics can download the model
# automatically. After that, the local .pt file is reused.
# =========================================================

MODEL_NAME = "yolo26n.pt"
CONFIDENCE = 0.45
CAMERA_INDEX = 0
WINDOW_NAME = "AI Object Detector"


def get_folder():
    """Return the folder where this script is stored."""
    return os.path.dirname(os.path.abspath(__file__))


def load_model():
    """Load a local model, or let Ultralytics download it."""
    folder = get_folder()
    local_model = os.path.join(folder, MODEL_NAME)

    try:
        if os.path.exists(local_model):
            print("Loading local model:", local_model)
            return YOLO(local_model)

        print("Model file not found locally.")
        print("Ultralytics will download", MODEL_NAME, "on the first run.")
        print("An internet connection is required only for this download.")
        return YOLO(MODEL_NAME)

    except Exception as e:
        print("\nCould not load the AI model.")
        print("Make sure the setup completed successfully and")
        print("that this computer has internet access for the first run.")
        print("\nError:", e)
        sys.exit(1)


def open_camera():
    """Open the default webcam."""
    camera = cv2.VideoCapture(CAMERA_INDEX)

    if not camera.isOpened():
        print("\nCould not open the webcam.")
        print("Check Windows camera permissions and make sure another")
        print("application is not exclusively using the camera.")
        sys.exit(1)

    return camera


def draw_fps(frame, fps):
    text = "FPS: " + str(round(fps, 1))
    cv2.putText(
        frame,
        text,
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )


def main():
    print("=" * 55)
    print("        AI OBJECT DETECTOR - PROTOTYPE")
    print("=" * 55)
    print("Press Q to quit.")
    print()

    model = load_model()
    camera = open_camera()

    previous_time = time.time()

    while True:
        success, frame = camera.read()

        if not success:
            print("Could not read a frame from the webcam.")
            break

        # Run object detection
        results = model.predict(
            source=frame,
            conf=CONFIDENCE,
            verbose=False
        )

        # Draw detections
        annotated_frame = results[0].plot()

        current_time = time.time()
        elapsed = current_time - previous_time

        if elapsed > 0:
            fps = 1 / elapsed
        else:
            fps = 0

        previous_time = current_time
        draw_fps(annotated_frame, fps)

        cv2.imshow(WINDOW_NAME, annotated_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()
    print("\nDetector stopped.")


if __name__ == "__main__":
    main()
