import cv2
img=cv2.imread('img.png')
cv2.namedWindow("1", 4)
cv2.imshow("1", img)

r= cv2.selectROI('ROI',img,False,False)
#x,y,w,h=r
print(r)
imCrop = img[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]

cv2.namedWindow("cut_image", 2)
cv2.resizeWindow("cut_image", 300, 300)
cv2.imshow("cut_image", imCrop)

cv2.waitKey(0)
cv2.destroyAllWindows()
