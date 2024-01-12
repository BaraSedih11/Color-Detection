import cv2
import numpy as np

cap = cv2.VideoCapture(0)

# # Define red color range in HSV format
# lower_red = np.array([0, 130, 130])
# upper_red = np.array([10, 200, 200])
# Define the lower and upper bounds for a very narrow red color range
lower_red = np.array([0, 200, 200])
upper_red = np.array([3, 255, 255])


# Initialize variables for pen path tracking and drawing
prev_point = None
drawing_canvas = np.zeros((480, 640, 3), dtype=np.uint8)

while True:
    # Get the current frame and flip it horizontally
    ret, frame = cap.read()
    frame_flipped = cv2.flip(frame, 1)

    # Convert the frame to HSV format and create a binary mask of the red pen
    hsv_frame = cv2.cvtColor(frame_flipped, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_frame, lower_red, upper_red)

    # Find the contour of the red pen
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=cv2.contourArea)  # Find contour with maximum area
        x, y, w, h = cv2.boundingRect(cnt)
        curr_point = (int(x + w/2), int(y + h/2))

        # Draw a line between the current and previous points using cv2.line()
        if prev_point is not None:
            cv2.line(drawing_canvas, prev_point, curr_point, (0, 0, 255), thickness=2)

        # Update the previous_point for the next iteration
        prev_point = curr_point

    # Add the drawing canvas to the original frame to visualize it
    result = cv2.addWeighted(frame_flipped, 1, drawing_canvas, 1, 0)

    # Show the result and print the mask
    cv2.imshow('frame', result)
    #cv2.imshow('mask', mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
