import cv2
from ultralytics import YOLO
import numpy as np
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import time

# Email Configuration
SENDER_EMAIL = "criminaldetectionheadquater@gmail.com"  # Replace with your email address
SENDER_PASSWORD = "lhck zrxy qera ztqa"  # Replace with your app password
RECIPIENT_EMAILS = ["sagnikb500@gmail.com", "ishitapanja17@gmail.com"]  # List of recipient emails

# Function to send email notifications with an image attachment
def send_email_notification(detected_info, image_path):
    """
    Send an alert email notification with an attached image.
    :param detected_info: String with detection details
    :param image_path: Path to the image file to be attached
    """
    subject = "Criminal Detection Alert"
    body = f"Detection Update:\n\n{detected_info}"

    # Set up email message
    msg = MIMEMultipart()
    msg['From'] = SENDER_EMAIL
    msg['To'] = ", ".join(RECIPIENT_EMAILS)
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    # Attach the image
    try:
        with open(image_path, 'rb') as img_file:
            img = MIMEImage(img_file.read())
            img.add_header('Content-Disposition', f'attachment; filename="detection.jpg"')
            msg.attach(img)
    except Exception as e:
        print(f"Error attaching image: {e}")

    # Send the email
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()  # Secure the connection
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, RECIPIENT_EMAILS, msg.as_string())
        print(f"Email notification sent: {detected_info}")
    except Exception as e:
        print(f"Error sending email: {e}")

# Load the trained YOLOv8 model
try:
    model = YOLO(r"C:\Users\ISHITA\Downloads\project17\backend\best.pt")  # Path to trained model weights
except Exception as e:
    print("Error loading YOLO model:", e)
    exit(1)

# Class names based on your model
CLASS_NAMES = ["criminal", "non-criminal"]

def detect_criminals_on_image(frame):
    """
    Detect criminals in a single image frame.
    :param frame: The input image frame (numpy array)
    :return: Annotated frame with detection boxes and labels
    """
    results = model.predict(frame, verbose=False)
    
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Bounding box coordinates
        scores = result.boxes.conf.cpu().numpy()  # Confidence scores
        classes = result.boxes.cls.cpu().numpy().astype(int)  # Class IDs

        for box, score, class_id in zip(boxes, scores, classes):
            label = f"{CLASS_NAMES[class_id]}: {score:.2f}"
            color = (0, 0, 255) if CLASS_NAMES[class_id] == "criminal" else (0, 255, 0)

            # Draw bounding box and label
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
            cv2.putText(frame, label, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Save the frame as an image file
            timestamp = int(time.time())
            image_path = f"detected_image_{timestamp}.jpg"
            cv2.imwrite(image_path, frame)

            # Send email notification
            detected_info = f"{CLASS_NAMES[class_id]} detected with confidence {score:.2f}."
            send_email_notification(detected_info, image_path)
    
    return frame

def run_detection_on_webcam():
    """
    Run criminal detection on webcam feed.
    Capture a single image on key press, detect, and show result.
    """
    cap = cv2.VideoCapture(0)  # Initialize webcam

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 'c' to capture an image for detection, or 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
        # Display the live feed
        cv2.imshow("Webcam - Press 'c' to Capture, 'q' to Quit", frame)

        # Capture or quit based on key press
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):  # Capture image for detection
            print("Capturing image for criminal detection...")
            annotated_frame = detect_criminals_on_image(frame)
            cv2.imshow("Detected Image", annotated_frame)
            cv2.waitKey(0)  # Pause to display result until a key is pressed

        elif key == ord('q'):  # Exit on 'q' press
            print("Exiting...")
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_detection_on_webcam()