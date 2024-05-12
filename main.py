import cv2

# Load the pre-trained Haar cascades for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
    return image

def main():
    # Option to choose between image or video stream
    choice = input("Choose an option:\n1. Image\n2. Video Stream\nEnter your choice (1/2): ")

    if choice == '1':
        # Read the image
        image_path = input("Enter the path to the image: ")
        image = cv2.imread(image_path)
        
        # Detect faces in the image
        faces_detected = detect_faces(image)
        
        # Display the image with detected faces
        cv2.imshow("Faces Detected", faces_detected)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    elif choice == '2':
        # Capture video from the default camera
        cap = cv2.VideoCapture(0)
        
        while True:
            # Read a frame from the video stream
            ret, frame = cap.read()
            
            # Detect faces in the frame
            frame_with_faces = detect_faces(frame)
            
            # Display the frame with detected faces
            cv2.imshow("Video Stream", frame_with_faces)
            
            # Press 'q' to exit the video stream
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release the video capture object
        cap.release()
        cv2.destroyAllWindows()
        
    else:
        print("Invalid choice. Please choose either 1 or 2.")

if __name__ == "__main__":
    main()
