import cv2
import numpy as np
from scipy.interpolate import splprep, splev
from skimage.morphology import skeletonize

# Load the image
image = cv2.imread('withLines4.png')

blurred = cv2.GaussianBlur(image, (25, 25), 0)
# Convert to HSV color space
hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

# Define the color range for the pink line
lower_color = np.array([150, 50, 50])   # Adjust based on the line color
upper_color = np.array([170, 255, 255])

# Create a mask for the pink line
mask = cv2.inRange(hsv, lower_color, upper_color)
# Skeletonize the cleaned mask to get the centerline
skeleton = skeletonize(mask // 255)  # Convert to binary and skeletonize
skeleton = (skeleton.astype(np.uint8)) * 255  # Convert back to 0-255 range
# Find contours directly on the mask
contours, _ = cv2.findContours(skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

if contours:
    # Assume the largest contour corresponds to the pink line
    largest_contour = max(contours, key=cv2.contourArea).squeeze()

    # Check if the contour is valid
    if largest_contour.ndim == 2 and largest_contour.shape[1] == 2:
        # Extract x and y coordinates of the contour
        x = largest_contour[:, 0]
        y = largest_contour[:, 1]

        # Parameterize the points using a normalized parameter
        t = np.linspace(0, 1, len(x))

        # Fit a spline to the x and y coordinates
        tck, _ = splprep([x, y], u=t, s=0)
        u_fine = np.linspace(0, 1, 1000)
        x_new, y_new = splev(u_fine, tck)

        # Compute curvature
        dx, dy = splev(u_fine, tck, der=1)
        d2x, d2y = splev(u_fine, tck, der=2)
        curvatures = np.abs(dx * d2y - dy * d2x) / (dx**2 + dy**2)**1.5
        normalized_curvatures = (curvatures - np.min(curvatures)) / (np.max(curvatures) - np.min(curvatures))
        average_curvature_score = np.mean(normalized_curvatures)
        print(f"Average curvature score (0 to 1): {average_curvature_score}")
        max_curvature = np.max(curvatures)
        print(f"Maximum curvature: {max_curvature}")

        # Draw the fitted spline on the image
        for i in range(len(x_new) - 1):
            pt1 = (int(x_new[i]), int(y_new[i]))
            pt2 = (int(x_new[i+1]), int(y_new[i+1]))
            cv2.line(image, pt1, pt2, (255, 0, 0), 2)
    else:
        print("Contour format is not valid.")
else:
    print("No contours found.")

# Display the result
cv2.imshow('Simplified Line Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
