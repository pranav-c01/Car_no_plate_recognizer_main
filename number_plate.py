import cv2
import time
import  os

harcascade = "model/haarcascade_russian_plate_number.xml"

#cap = cv2.VideoCapture(0)

#cap.set(3, 640) # width
#cap.set(4, 480) #height

min_area = 500
list_images = os.listdir('images')
i=len(list_images)-1

while i>=0:
#    success, img = cap.read()

    plate_cascade = cv2.CascadeClassifier(harcascade)
    img = cv2.imread(f"images\{list_images[i]}")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#    img_gray = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),(300,100),interpolation=cv2.INTER_AREA)
#    cv2.imshow("img",img_gray)
#   print(img_gray.shape,img.shape)

    plates = plate_cascade.detectMultiScale(img_gray, 1.2)
    print(plates)
    for (x,y,w,h) in plates:
        #area = w * h
        #print(area)
        if 1>0:
            cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(img, "Number Plate", (x,y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)

            img_roi = img[y: y+h, x:x+w]
            #cv2.imshow("ROI", img_roi)


    cv2.namedWindow("Result", cv2.WINDOW_NORMAL) 
    cv2.resizeWindow("Result", 500, 500) 
    cv2.imshow("Result", img)
    count = time.time()
    
    while True:

	    if cv2.waitKey(1) & 0xFF == ord('s'):
        	cv2.imwrite("plates/scaned_img_" + str(count)[-1:-6:-1] + ".jpg", img_roi)
	        cv2.rectangle(img, (0,200), (640,300), (0,255,0), cv2.FILLED)
        	cv2.putText(img, "Plate Saved", (150, 265), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)
	        cv2.imshow("Results",img)
        	cv2.waitKey(500)
	        cv2.destroyAllWindows()
	        break
    
    i-=1


