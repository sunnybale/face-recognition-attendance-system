import cv2

print("Opening camera...")
cap = cv2.VideoCapture(0)

print("Camera opened:", cap.isOpened())

if not cap.isOpened():
    print("❌ Camera not authorized or busy. macOS may be blocking access.")
    exit()

while True:
    ret, frame = cap.read()
    print("Frame received:", ret)

    if not ret:
        break

    cv2.imshow("Camera Test - Press ESC to exit", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
