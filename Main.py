import cv2

cap = cv2.VideoCapture("assets/traffic.mp4")
# cap = cv2.VideoCapture(0)

while True:
    _, frame1 = cap.read()
    _, frame2 = cap.read()

    diff = cv2.absdiff(frame1, frame2)
    grayscale = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grayscale, (5, 5), 0)
    _, threshold = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(threshold, None, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(frame1, contours, -1, (0, 255, 0), 1)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if cv2.contourArea(c) < 1000 or w > 200:
            continue
        cv2.rectangle(frame1, (x,y), (x+w, y+h), (0, 255, 0), 2)
        # cv2.putText(frame1, str(w), (x,y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (255,255,0), 1)

    cv2.imshow("Move Detection", frame1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()