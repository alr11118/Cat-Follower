import cv2
from ultralytics import YOLO

def findCenterOfBox(xmin, ymin, xmax, ymax):
    width = xmax-xmin
    height = ymax-ymin
    midPoint = (xmin+width//2, ymin+height//2)
    return midPoint

def main():
    # Load the image
    image = cv2.imread("data/poncik.jpg")
    
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
    # Process resaults
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Extract coordinates
            # xyxy format: [xmin, ymin, xmax, ymax]
            xmin, ymin, xmax, ymax = map(int, box.xyxy[0])
            
            # Get confidence score
            confidence = float(box.conf[0])
            
            print(f"Cat detected with {confidence:.2f} confidence.")
            print(f"Coordinates: Top-Left ({xmin}, {ymin}), Bottom-Right ({xmax}, {ymax})")
            midPoint = findCenterOfBox(xmin, ymin, xmax, ymax)
            print(f"Midpoint of bounding box({midPoint})")

    # Save or display the result
    cv2.imwrite("data/detected_cat.jpg", image)

    # Show or save the results
    results[0].show()

if __name__ == "__main__":
    main()
