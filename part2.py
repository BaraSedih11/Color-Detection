import cv2
import numpy as np

def calculate_distance(point1, point2):
    return np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

# Define colors in HSV space
green_lower = np.array([50, 100, 100])
green_upper = np.array([80, 255, 255])

# Initialize variables
finger_positions = []
zoom_factor = 1.0

# Initialize camera capture
cap = cv2.VideoCapture(0)

while True:
    # Read frame from camera
    ret, frame = cap.read()

    # Flip the frame horizontally for mirror effect
    frame = cv2.flip(frame, 1)

    # Convert frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a mask for green color
    mask = cv2.inRange(hsv, green_lower, green_upper)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find two largest contours (assuming they represent the fingers)
    largest_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]

    # Get finger positions
    finger_positions = []
    for contour in largest_contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            finger_positions.append((cx, cy))

    # Perform zoom in/out based on finger positions
    if len(finger_positions) == 2:
        finger1, finger2 = finger_positions
        distance = calculate_distance(finger1, finger2)
        if distance > 0:
            zoom_factor = 1.0 + (distance / 200)  # Adjust the scaling factor as needed

    # Determine the new dimensions of the frame after scaling
    new_height = int(frame.shape[0] * zoom_factor)
    new_width = int(frame.shape[1] * zoom_factor)
    new_size = (new_width, new_height)

    # Apply zooming transformation to the frame with new_size parameter passed
    scaled_frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_LINEAR)


    # Display the original and scaled frames
    cv2.imshow('Original', frame)
    cv2.imshow('Scaled', scaled_frame)

    # Check for 'q' key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy windows
cap.release()
cv2.destroyAllWindows()
