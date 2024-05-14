import cv2
import pafy
from ultralytics import YOLO

# Initialize a YOLO-World model
model = YOLO('models/yolov8s-world.pt')  # or choose yolov8m/l-world.pt

# model.to('mps')
# Define custom classes
model.set_classes(["car", "bus", "building", "traffic light"])

# Get the best video URL
url = "https://www.youtube.com/embed/rXlaAwtXUlY?si=zKdCB7BJUBz8WcJU"
video = pafy.new(url).getbestvideo(preftype="mp4")
if video is None:
    print("No video found at the provided URL.")
else:
    results = model.track(source=video.url, conf=0.3, iou=0.5, show=True)

    # Add functionality to end the program when 'q' is pressed
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()