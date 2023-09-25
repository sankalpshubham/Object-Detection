import cv2
from random import randrange

trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')         # loading pre-trained model data on face frontals from opencv
img = cv2.imread('elon_musk.jpg')                                                       # choosing an image to detect faces in
img = cv2.resize(img, (700, 890))
# img = cv2.imread('two_ppl.png') 
# print("WOOOOOOOOOOW", img)
# img = cv2.resize(img, (1000, 520))
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face_coordinates = trained_face_data.detectMultiScale(img_gray)                         # detecting faces model           

# -------- reading image ----------
# for (x, y, w, h) in face_coordinates:                                                   # assigning the face_coords [[x y w h]] to the 4-tuple (each face is an individual idx)
#     cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)                              # draw rectangle the faces (color of the rec, 2 is thickness of the rec) (randrange(256) for random box color)

# cv2.imshow("Face Detector Window", img)
# cv2.waitKey(0)


# -------- reading video ----------
#webcam = cv2.VideoCapture('impression_vid.mp4')                                        # capturing faces from a recorded video
webcam = cv2.VideoCapture(0)                                                            # capturing video from camera (0 is default cam input)

while True:                                                                             # running the while look forever until the vid ends or we kill the webcam
    successful_frame_read, frame = webcam.read()                                        # reading from webcam | returns T/F if read or not, and the actual image/frame
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    vid_face_coordinates = trained_face_data.detectMultiScale(frame_gray)

    for (x, y, w, h) in vid_face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Webcam Face Detector Window", frame)
    key = cv2.waitKey(1)                                                                      # program automatically presses a key to continue/execute reading in frame (1ms)

    if key == 81 or key == 113:
        break

webcam.release()


print("Completed!")