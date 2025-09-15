# HLB-FI-6AxME

# Projektidee

Automatisches Garagentor (o.ä.): Embedded System mit Kamera erkennt Kennzeichen (und/oder RFID) und öffnet ggf. ein Garagentor / Schranken.

Bilderkennung z.B. mit OpenCV https://opencv.org/

## Benötigte Hardware (Variante 1: Raspberry Pi)

- Raspberry Pi (Version mit OpenCV kompatibel)
- Camera Modul für den Raspberry (USB oder Shield?)
- RFID Reader
- Ansteuerung von Aktoren (Servo? Motor?)
- Optional: Display

Links:
- https://opencv.org/blog/raspberry-pi-with-opencv/

## Benötigte Hardware (Variante 2: Microcontroller+Kamera - Applikation - Microcontroller+Aktor)

- Microcontroller (IoT fähig) + Kamera, z.B. ESP32-Cam
- Microcontroller (IoT fähig) + Aktoren, z.B. ESP32, Raspberry Pi Pico, ...
- RFID Reader
- WiFi
- "Server" für die Applikation
- Optional: Display

## Benötigte Software (beide Versionen)

- Bilderkennung: OpenCV + Python
- 


# Erste Schritte

Installation Python (https://www.python.org/)
Installation OpenCV für Python `pip install opencv-python`
(Anmerkung: `pip` ist der _Package Manager_ für Python, dient dazu Bibliotheken zu installieren)

Beispiel-Programm (Quelle: https://opencv.org/blog/raspberry-pi-with-opencv/)

```python
import cv2
import time
 
# Initialize video capture
capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 576)
capture.set(cv2.CAP_PROP_FPS, 30)  # Requesting 30 FPS from the camera
 
# Background subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2()
 
# Current mode
mode = "normal"
mode = "bg_sub"  # Default mode for demonstration
 
# FPS calculation
prev_time = time.time()
 
# Video writer initialization with explicit FPS
save_fps = 15.0  # Adjust this based on your actual processing speed
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter("output.avi", fourcc, save_fps, (1024, 576))
 
while True:
    ret, frame = capture.read()
    if not ret:
        break
 
    frame = cv2.flip(frame, 1)
    display_frame = frame.copy()
 
    if mode == "threshold":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, display_frame = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        display_frame = cv2.cvtColor(display_frame, cv2.COLOR_GRAY2BGR)
 
    elif mode == "edge":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        display_frame = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
 
    elif mode == "bg_sub":
        fg_mask = bg_subtractor.apply(frame)
        display_frame = cv2.bitwise_and(frame, frame, mask=fg_mask)
 
    elif mode == "contour":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        display_frame = cv2.drawContours(frame.copy(), contours, -1, (0, 255, 0), 2)
 
    # Calculate actual processing FPS
    curr_time = time.time()
    processing_fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
 
    # Display actual processing FPS
    cv2.putText(
        display_frame, f"FPS: {int(processing_fps)} Mode: {mode}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
    )
 
    # Write frame to video
    out.write(display_frame)
 
    # Show video
    cv2.imshow("Live Video", display_frame)
 
    key = cv2.waitKey(1) & 0xFF
    if key == ord("t"):
        mode = "threshold"
    elif key == ord("e"):
        mode = "edge"
    elif key == ord("b"):
        mode = "bg_sub"
    elif key == ord("c"):
        mode = "contour"
    elif key == ord("n"):
        mode = "normal"
    elif key == ord("q"):
        break
 
# Clean up
capture.release()
out.release()
cv2.destroyAllWindows()
```

Aktuelles Programm mit Ziffern-Bounding-Boxen:

```python
import cv2
import time
import numpy as np

# Source: https://www.mindee.com/blog/digit-recognition-python-opencv
def extract_digits_from_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    digit_regions = []
    bounding_boxes = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h > 10 and w > 5:
            roi = thresh[y:y+h, x:x+w]
            resized = cv2.resize(roi, (20, 20))
            digit_regions.append(resized)
            bounding_boxes.append((x, y, w, h))

    # Sort digits left to right
    sorted_digits = [x for _, x in sorted(zip(bounding_boxes, digit_regions), key=lambda b: b[0][0])]
    return sorted_digits, bounding_boxes

def recognize_digits(digit_images, knn_model):
    results = []
    for digit_img in digit_images:
        sample = digit_img.reshape((1, 400)).astype(np.float32)
        ret, result, _, _ = knn_model.findNearest(sample, k=5)
        results.append(int(result[0][0]))
    return results

# Train Picture NN
img = cv2.imread('Download.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Split into individual digit cells (20x20 pixels)
cells = [np.hsplit(row, 100) for row in np.vsplit(gray, 50)]
x = np.array(cells)

train = x[:, :50].reshape(-1, 400).astype(np.float32)
test = x[:, 50:100].reshape(-1, 400).astype(np.float32)

# Labels: 250 of each digit 0–9
labels = np.repeat(np.arange(10), 250)[:, np.newaxis]
train_labels = labels.copy()
test_labels = labels.copy()

# Create and train KNN model
knn = cv2.ml.KNearest_create()
knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)

ret, result, neighbours, dist = knn.findNearest(test, k=5)

accuracy = (result == test_labels).mean() * 100
print(f"Test accuracy: {accuracy:.2f}%")



# Initialize video capture
capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 576)
capture.set(cv2.CAP_PROP_FPS, 30)  # Requesting 30 FPS from the camera
 
# Background subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2()
 
# Current mode
mode = "normal"
 
# FPS calculation
prev_time = time.time()
 
# Video writer initialization with explicit FPS
#save_fps = 15.0  # Adjust this based on your actual processing speed
#fourcc = cv2.VideoWriter_fourcc(*"XVID")
#out = cv2.VideoWriter("output.avi", fourcc, save_fps, (1024, 576))
 
while True:
    ret, frame = capture.read()
    if not ret:
        break
#    frame = cv2.flip(frame, 1)

    digits, boxes = extract_digits_from_image(frame)
    predictions = recognize_digits(digits, knn)
    print("Detected digits:", predictions)

    display_frame = frame.copy()
 
    for (x, y, w, h), digit in zip(boxes, predictions):
        cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(display_frame, str(digit), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    if mode == "threshold":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, display_frame = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        display_frame = cv2.cvtColor(display_frame, cv2.COLOR_GRAY2BGR)
 
    elif mode == "edge":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        display_frame = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
 
    elif mode == "bg_sub":
        fg_mask = bg_subtractor.apply(frame)
        display_frame = cv2.bitwise_and(frame, frame, mask=fg_mask)
 
    elif mode == "contour":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        display_frame = cv2.drawContours(frame.copy(), contours, -1, (0, 255, 0), 2)
 
    # Calculate actual processing FPS
    curr_time = time.time()
    processing_fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
 
    # Display actual processing FPS
    cv2.putText(
        display_frame, f"FPS: {int(processing_fps)} Mode: {mode}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
    )
 
    # Write frame to video
    #out.write(display_frame)
 
    # Show video
    cv2.imshow("Live Video", display_frame)
 
    key = cv2.waitKey(1) & 0xFF
    if key == ord("t"):
        mode = "threshold"
    elif key == ord("e"):
        mode = "edge"
    elif key == ord("b"):
        mode = "bg_sub"
    elif key == ord("c"):
        mode = "contour"
    elif key == ord("n"):
        mode = "normal"
    elif key == ord("q"):
        break
 
# Clean up
capture.release()
#out.release()
cv2.destroyAllWindows()
```



