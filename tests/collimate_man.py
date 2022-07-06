import cv2
import numpy as np


class Collimate:
    def __init__(self, img, windowname="Select point", points=None):
        self.img = img
        self.windowname = windowname
        self.cache = None
        self.curr_pt = []
        self.point   = []
        
    def draw_marker(self, event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            self.cache = self.img.copy()
            cv2.drawMarker(self.img,(x,y),(255,0,0), cv2.MARKER_CROSS, markerSize = 3)
            ix,iy = x,y
            return [ix,iy]

    def select_one(self,event,x,y):
        cv2.namedWindow(self.windowname, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.windowname, self.draw_marker)
        
        while(1):
            cv2.namedWindow(self.windowname,cv2.WINDOW_NORMAL)
            cv2.imshow(self.windowname,self.img)
            k = cv2.waitKey(20) & 0xFF
            if k == 27:
                cv2.destroyAllWindows()
                break
            elif k == ord('p'):
                '''print current point coordinates'''
                print(self.point[-1])
                
            # elif k == ord('d'):
            #     self.img = self.cache.copy()
            #     cv2.imshow(self.windowname, self.img)
            #     self.cache = self.img.copy()
            #     self.point.append([x,y])        
            #     cv2.drawMarker(img,(x,y),(255,0,0), cv2.MARKER_CROSS, markerSize = 3)
                           
    def __getitem__(self):
        return self.point  

                           
    # def getpt(self,count=1,img=None):
    #     if img is not None:
    #         self.img = img
    #     else:
    #         self.img = self.img1.copy()
    #     cv2.namedWindow(self.windowname,cv2.WINDOW_NORMAL)
    #     cv2.imshow(self.windowname,self.img)
    #     cv2.setMouseCallback(self.windowname,self.select_point)
    #     self.point = []
    #     while(1):
    #         cv2.imshow(self.windowname,self.img)
    #         k = cv2.waitKey(20) & 0xFF
    #         if k == 27 or len(self.point)>=count:
    #             break
    #         #print(self.point)
    #     cv2.setMouseCallback(self.windowname, lambda *args : None)
    #     #cv2.destroyAllWindows()
    #     return self.point, self.img

        def __getitem__(self):
            return self.point  


if __name__=='__main__':
    
    img = cv2.imread('data/img/L/DJI_0832.tif')
    
    windowname = 'image'
    coordinateStore = Collimate.select_one(img, windowname)



    # pts,img = coordinateStore.getpt(1)
    # print(pts)

    # pts,img = coordinateStore.getpt(1,img)
    # print(pts)

    # cv2.imshow(windowname,img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    
    
# class ManualMatcher:
#     def __init__(self, image, window_name="Select point", points=None):
#         import pdb; pdb.set_trace()
#         self.image = image
#         self.window_name = window_name
#         self.points = points
#         # self.img_idx = 0    # 0: left; 1: right
#         cv2.namedWindow(window_name)
#         cv2.setMouseCallback(window_name, self.onMouse)
    
#     def onMouse(self, event, x, y, flags, userdata):
#         if event == cv2.EVENT_LBUTTONDOWN:
#             self.points[self.img_idx].append((x, y))
        
#         elif event == cv2.EVENT_LBUTTONUP:
#             # Switch images in a ping-pong fashion
#             if len(self.points[0]) != len(self.points[1]):
#                 self.img_idx = 1 - self.img_idx
    
#     def run(self):
#         print ("Select your matches. Press SPACE when done.")
#         while True:
#             cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
#             cv2.imshow(self.window_name, self.image)

#             # cv2.imshow("finalImg",img0_und)
            
#             # img = cv2.drawKeypoints(self.images[self.img_idx], [cv2.KeyPoint(p[0],p[1], 7.) for p in self.points[self.img_idx]], color=(0,0,255))
#             key = cv2.waitKey(50) & 0xFF
#             if key == ord(' '): break    # finish if SPACE is pressed
    