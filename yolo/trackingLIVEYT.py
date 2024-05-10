import cv2
from ultralytics import YOLO

# Initialize a YOLO-World model
model = YOLO('yolov8s-world.pt')  # or choose yolov8m/l-world.pt

# model.to('mps')
# Define custom classes
model.set_classes(["eyes", "ears", "nose", "mouth", "hair"])

results = model.predict(source=0, conf=0.3, iou=0.5, show=True)

# Add functionality to end the program when 'q' is pressed
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

