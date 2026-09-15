import cv2
from ultralytics import YOLO

# Load the image
image = cv2.imread("data/minnak.png")
 
# Get the original dimensions
original_height, original_width = image.shape[:2]
 
# Define new width while maintaining the aspect ratio
new_width = 512
aspect_ratio = new_width / original_width
new_height = int(original_height * aspect_ratio)  # Compute height based on aspect ratio
 
# Resize the image
resized_image = cv2.resize(image, (new_width, new_height))
 
# Display the resized image
#cv2.imshow("Resized Image", resized_image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
cv2.imwrite('data/resizedImage.jpg', resized_image)

# Run the model on it
# Load the pre-trained YOLO model
model = YOLO("yolo11s.pt")

# Predict and filter for class 15 (cat)
results = model.predict("data/resizedImage.jpg", classes=[15])

# Show or save the results
results[0].show()
