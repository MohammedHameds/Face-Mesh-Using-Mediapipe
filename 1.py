import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh.FaceMesh()

img = cv2.imread('image.jpg',1)
img = cv2.resize(img,(450,550))
rbg_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

result = mp_face_mesh.process(rbg_image)

height,width,_ = img.shape

for facial_landmarks in result.multi_face_landmarks :
   for  i in range(0,468) :
    pt1 = facial_landmarks.landmark[i]
    x = int(pt1.x * width)
    y = int(pt1.y * height)
    
    cv2.circle(img,(x,y),2,(255,0,0),-1)


cv2.imshow('Original image',img)
cv2.imwrite('output.png',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

