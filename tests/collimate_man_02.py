import cv2
import numpy as np

ix,iy = -1,-1
# mouse callback function
def draw_circle(event,x,y,flags,param):
    global ix,iy
    global cache
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cache = img.copy()
        cv2.drawMarker(img,(x,y),(255,0,0), cv2.MARKER_CROSS, markerSize = 3)
        ix,iy = x,y
        # return [ix,iy]

# Create a black image, a window and bind the function to window
img = cv2.imread('data/img/L/DJI_0832.tif')
pts = []
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('image',draw_circle)

while(1):
    cv2.imshow('image',img)
    k = cv2.waitKey(20) & 0xFF
    if k == 27:
        break
    elif k == ord('a'):
        print(ix,iy)
    elif k == ord('d'):
        # print(ix,iy)
        # cv2.destroyAllWindows()
        img = cache.copy()
        cv2.imshow('image', img)
        cv2.setMouseCallback('image', draw_circle)
        
cv2.destroyAllWindows()
