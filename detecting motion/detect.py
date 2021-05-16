import cv2, time

video = cv2.VideoCapture(0)

frame_1 = None

while True:
    check, frame = video.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if frame_1 is None:
        frame_1 = gray
        continue
    
    delta = cv2.absdiff(frame_1, gray)     ##calculate the diff between the 1st frame and 2nd frame.
    threshhold_d = cv2.threshhold(delta, 30 , 255, cv2.THRESH_BINARY)[1]
    threshhold_d = cv2.dilate(threshhold_d, None, iterations = 10)

    (cnts,_) = cv2.findCountours(threshhold_d.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #Used (), because it needs touple values

    for countour in cnts:
        if cv2.countourArea(countour) < 1100:
            continue
    
        (x, y, w, h) = cv2.boundingRect(countour) #values will be assigned automatically

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 250, 50), 3) #values from x,y,w,h will be used to draw the rectangle


    cv2.imshow("Gray", gray)
    cv2.imshow("Delta", delta)
    cv2.imshow("Threshhold", threshhold_d)
    cv2.imshow("ColorFrame", frame)

    key = cv2.waitKey(1)
    print(gray)

    if key == ord("q"):
            break
    
    #threshhold_gray = cv2.threshhold(frame_1, 30 , 255, cv2.THRESH_BINARY)

video.release()
cv2.destroyAllWindows