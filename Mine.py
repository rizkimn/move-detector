import cv2

cap = cv2.VideoCapture("assets/traffic.mp4")
# cap = cv2.VideoCapture(0)
density = False
while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    _, frame2 = cap.read()
    frame2 = cv2.flip(frame2, 1)

    diff = cv2.absdiff(frame, frame2)
    grayscale = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grayscale, (5,5), 0)

    _, threshold = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(threshold, None, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if density:
        color = (0,0,255)
    else:
        color = (0,255,0)

    total = 0

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if cv2.contourArea(c) < 1000 or w > 140:
            continue

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        # cv2.putText(frame, str(w), (x, y+h+20), cv2.FONT_HERSHEY_DUPLEX, 1, color, 1, cv2.LINE_AA)
        total += 1

    cv2.putText(frame, str(total), (10, 30), cv2.FONT_HERSHEY_DUPLEX, 1, color, 1, cv2.LINE_AA)
    cv2.imshow("My System", frame)


    if total > 12:
        density = True
    else:
        density = False

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()