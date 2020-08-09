#python script that captures images from the webcam video stream
#extract all faces from the image_frame (using haarcascades)
#stores the face information into numpy array

#Step1 : read and show video stream , capture images
#Step2: detect faces and show bounding box
#Step3: flatten the largest face image and save in a numpy array
#Step4: Repeat the above for multiple people to genrate training data

import cv2
import numpy as np

#Init camera
cap = cv2.VideoCapture(0)

#for face detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

skip = 0
face_data = []
dataset_path = './data/'
file_name = input("Enter the name of the person : ")

while True:
    #ret is a boolean regarding whether or not there was a return at all,
    #at the frame is each frame that is returned. If there is no frame, 
    #you wont get an error, you will get None.
    ret, frame = cap.read()
    
    if ret == False:
        continue
        
        
    #need to store the gray frame
    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(frame,1.3,5) #detect multiple faces
    #print(faces)

    if len(faces)==0:
        continue
  
    #we are going to store on the basis of face ares 
    #i.e. in case of x,y,w,h we are going to sort on basis of w and h (f[2] * f[3]) of each face i.e. area
    
    faces = sorted(faces, key = lambda f:f[2]*f[3])
    
    #pick the last face cause it has the largest face area 
    #we are iterating from the end
    for face in faces[-1:]:
        x,y,w,h = face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2) #we make rectangle arounf a face
        #Extract (crop out the req face): region of interest
        
        offset = 10 #adding a pad of 10 pixels around the face
        face_section = frame[y-offset:y+h+offset, x-offset: x+w+offset] #in face axis is y, x (not x,y)
        face_section  = cv2.resize(face_section,(100,100))
        
        skip += 1
        
        if(skip %10 == 0):
            #store the 10th face later
            face_data.append(face_section)
            print(len(face_data))
        
        
        
    cv2.imshow("Frame", frame)
    cv2.imshow("Face Section", face_section)
    
    
    key_pressed = cv2.waitKey(1) & 0xFF
    
    if key_pressed == ord('q'):
        break

# Convert our face list array into a numpy array
face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0],-1)) #the row should be same as the number of faces
#the col can fr figured out automatically
print(face_data.shape)

# Save this data into file system
np.save(dataset_path+file_name+'.npy',face_data)
print("Data Successfully save at "+dataset_path+file_name+'.npy')        
        
cap.release()
cv2.destroyAllWindows()
