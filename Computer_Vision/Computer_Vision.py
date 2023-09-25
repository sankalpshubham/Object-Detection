import cv2

classifier_file = "cars.xml"                                                            # pre-trained car classifier data
car_tracker = cv2.CascadeClassifier(classifier_file)

# img_file = "ny.jpg"
# img = cv2.imread(img_file)
# #img = cv2.resize(img, (1200, 820))
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# car_coordinates = car_tracker.detectMultiScale(img_gray)

# for (x, y, w, h) in car_coordinates:                                                   # assigning the face_coords [[x y w h]] to the 4-tuple (each face is an individual idx)
#     cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# cv2.imshow("Car Detector Window", img)
# cv2.waitKey(0)

# ------------ video ---------------
video = cv2.VideoCapture("tesla-car-cam.mp4")

while True:
    read_successful, frame = video.read()
    
    if read_successful:
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        print("error reading frame")
        break

    vid_car_coordinates = car_tracker.detectMultiScale(frame_gray)
    for (x, y, w, h) in vid_car_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Webcam Car Detector Window", frame)
    key = cv2.waitKey(1)                                                                      # program automatically presses a key to continue/execute reading in frame (1ms)

    if key == 81 or key == 113:
        break

video.release()

print("Code Completed!")