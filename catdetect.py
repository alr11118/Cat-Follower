import cv2
# Load the image
image = cv2.imread("input.jpeg")
 
# Get the original dimensions
original_height, original_width = image.shape[:2]
 
# Define new width while maintaining the aspect ratio
new_width = 300
aspect_ratio = new_width / original_width
new_height = int(original_height * aspect_ratio)  # Compute height based on aspect ratio
 
# Resize the image
resized_image = cv2.resize(image, (new_width, new_height))
 
# Display the resized image
cv2.imshow("Resized Image", resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()



# Run the model on it
# Return Weather cat is there