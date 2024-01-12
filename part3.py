import cv2
import numpy as np

OBJECT_DISTANCE_THRESHOLD = 25  # Adjust the threshold as per your requirement
# LOWER_BLUE = np.array([90, 50, 50])  # Adjust the lower range of blue color
# UPPER_BLUE = np.array([160, 255, 255])  # Adjust the upper range of blue color

# Define the lower and upper bounds for a narrower blue color range
LOWER_BLUE = np.array([110, 150, 150])
UPPER_BLUE = np.array([130, 255, 255])


def calculate_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def capture_image(frame, file_name):
    cv2.imwrite(file_name, frame)
    print("Image captured and stored as", file_name)
    cv2.imshow("Captured Image", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detect_blue_objects():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Failed to open camera")
        return

    object1_detected = False
    object1_position = (0, 0)
    object2_detected = False
    object2_position = (0, 0)

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to read frame")
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE)
        blurred = cv2.GaussianBlur(mask, (15, 15), 0)

        contours, _ = cv2.findContours(blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) > 300:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                if not object1_detected:
                    object1_position = (x + w // 2, y + h // 2)
                    object1_detected = True
                elif not object2_detected:
                    object2_position = (x + w // 2, y + h // 2)
                    object2_detected = True

        if object1_detected and object2_detected:
            distance = calculate_distance(object1_position[0], object1_position[1], object2_position[0], object2_position[1])
            if distance < OBJECT_DISTANCE_THRESHOLD:
                capture_image(frame, "blue_objects_combined.png")
            object1_detected = False
            object2_detected = False

        cv2.imshow("Blue Object Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

detect_blue_objects()
