import cv2
#import train as tt

img = cv2.imread("gand.jpg",0)
img=cv2.resize(img,(512,512))
#img=cv2.GaussianBlur(img,(5,5),0)
ret,thr=cv2.threshold(img,25,255,cv2.THRESH_BINARY)

_,con,_=cv2.findContours(thr,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
rects = [cv2.boundingRect(co) for co in con]
for rect in rects:
    # Draw the rectangles
    cv2.rectangle(img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
    # Make the rectangular region around the digit
    '''leng = int(rect[3] *2)
    pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
    pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
    roi = thr[pt1:pt1+leng, pt2:pt2+leng]
    break
    # Resize the image
    #roi = cv2.resize(roi, (15, 15))#, interpolation=cv2.INTER_AREA)
    #roi = cv2.dilate(roi, (5, 5))
    #tt.test_nn(tt.x,roi)'''
cv2.imshow("test",img)
cv2.waitKey(0)
cv2.destroyAllWindows()


#"Test0/gand.jpg"