import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh.FaceMesh()
cap = cv2.VideoCapture(0)

while cap.isOpened() :
    ret,frame = cap.read()
    height,width,_ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_face_mesh.process(rgb_frame)
    
    for facial_landmarks in results.multi_face_landmarks :
        for i in range (0,468):
            pt = facial_landmarks.landmark[i]
            x = int (width * pt.x)
            y = int (height* pt.y)
            cv2.circle(frame,(x,y),2,(255,0,0),-1)
    
    cv2.imshow('Frame',frame)
    if cv2.waitKey(27) == ord('q'):
        break


cv2.destroyAllWindows()
cap.release()    