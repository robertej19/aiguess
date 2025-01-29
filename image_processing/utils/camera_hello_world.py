import cv2

cap = cv2.VideoCapture('/dev/video0')

while True:
    ret, frame = cap.read()
    print(frame)
    if not ret:
        continue
    cv2.imshow('usb cam test', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
