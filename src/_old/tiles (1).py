import numpy as np
import cv2 
import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('qt5agg')
import os           
                
# impath = os.path.abspath('t0/img/IMG_0520.JPG')
# image = cv2.imread(impath, cv2.IMREAD_COLOR)
# x1=0
# y1=0
# M=1000
# N=2000 
# viz = True
# out_dir='tiles'
# overlap = 20
# plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), interpolation='nearest')
# plt.show()  
# colDivisor = 2
# rowDivisor = 2
# writeTile2Disk = True

def subdivideImage(image, rowDivisor=2, colDivisor=2, overlap = 200, viz=False, out_dir='tiles', writeTile2Disk=True):
    assert not (image is None), 'Invalid image input'
        
    image = image.astype('float32')
    H = image.shape[0]
    W = image.shape[1]    
    DY  = round(H/rowDivisor/10)*10;
    DX  = round(W/colDivisor/10)*10;
    dim = (rowDivisor,colDivisor)
    
    # Check image dimensions
    # if not W % colDivisor == 0:
    #     print('Number of columns non divisible by the ColDivisor. Removing last column.')
    #     image = image[:, 0:-1]
    # if not H % rowDivisor == 0:
    #     print('Number of rows non divisible by the RowDivisor. Removing last row')
    #     image = image[0:-1, :]
    
    tiles = [] 
    limits = []   
    for col in range(0, colDivisor):
        for row in range(0, rowDivisor):
            tileIdx = np.ravel_multi_index((row,col), dim, order='F');
            limits.append((max(0, col*DX - overlap), 
                           max(0, row*DY - overlap),
                           max(0, col*DX - overlap) + DX+overlap,
                           max(0, row*DY - overlap) + DY+overlap ) )
            # print(f'Tile {tileIdx}: xlim = ({ limits[tileIdx][0], limits[tileIdx][2]}), ylim = {limits[tileIdx][1], limits[tileIdx][3]}')
            tile = image[limits[tileIdx][1]:limits[tileIdx][3], 
                         limits[tileIdx][0]:limits[tileIdx][2] ]
            tiles.append(tile) #.astype('float32')
            if writeTile2Disk:
                isExist = os.path.exists(out_dir)
                if not isExist:
                    os.makedirs(out_dir)       
                cv2.imwrite(os.path.join(out_dir,'tile_'+str(tileIdx)+'_'
                         +str(limits[tileIdx][0])+'_'+str(limits[tileIdx][1])+'.jpg'), tile)

    return tiles, limits
       
def appendPred(pred, predTile):
    # keys0 = [ key for key in pred.keys() if 'descriptors' not in key]
    predNew = {k: np.append(pred[k], v, axis=0) for k, v in predTile.items() if 'descriptors' not in k}  
    predNew['descriptors0'] = np.append(pred['descriptors0'], predTile['descriptors0'], axis=1)
    predNew['descriptors1'] = np.append(pred['descriptors1'], predTile['descriptors1'], axis=1)
    return predNew   

def applyMatchesOffset(matches, offset):
    for i, mtch in enumerate(matches):
        if matches[i] > -1: 
            matches[i] = mtch + offset
            
    return matches  


# impath = os.path.abspath('t0/img/IMG_0520.JPG')
# image = cv2.imread(impath, cv2.IMREAD_COLOR)
# x1=0
# y1=0
# M=1000
# N=2000 
# viz = True
# out_dir='tiles'
# overlap = 200

# def subdivideImage(image, x1=0, y1=0, M=1000, N=2000, viz=False, out_dir='tiles'):
#         # if image is None:
#     #     print('Invalid image input')
#     #     return None, None
#     assert not (image is None), 'Invalid image input'
    
#     isExist = os.path.exists(out_dir)
#     if not isExist:
#         os.makedirs(out_dir)

    
#     image_copy = image.copy()
#     imgheight = image.shape[0]
#     imgwidth = image.shape[1]
       
#     tiles = [] 
#     limits = []
    
#     for y in range(0, imgheight, M):
#         for x in range(0, imgwidth, N):
#             if (imgheight - y) < M or (imgwidth - x) < N:
#                 break
                
#             y1 = y + M
#             x1 = x + N
    
#             # check whether the patch width or height exceeds the image width or height
#             if x1 >= imgwidth and y1 >= imgheight:
#                 x1 = imgwidth - 1
#                 y1 = imgheight - 1
#                 #Crop into patches of size MxN
#                 tile = image_copy[y:y+M, x:x+N]
#                 #Save each patch into file directory
#                 cv2.imwrite(os.path.join(out_dir,'tile_'+str(x)+'_'+str(y)+'.jpg'), tile)
#                 tiles.append(tile)
#                 limits.append((x, y, x1, y1))
                
#             elif y1 >= imgheight: # when patch height exceeds the image height
#                 y1 = imgheight - 1
#                 #Crop into patches of size MxN
#                 tile = image_copy[y:y+M, x:x+N]
#                 #Save each patch into file directory
#                 cv2.imwrite(os.path.join(out_dir,'tile_'+str(x)+'_'+str(y)+'.jpg'), tile)
#                 tiles.append(tile)
#                 limits.append((x, y, x1, y1))

#             elif x1 >= imgwidth: # when patch width exceeds the image width
#                 x1 = imgwidth - 1
#                 #Crop into patches of size MxN
#                 tile = image_copy[y:y+M, x:x+N]
#                 #Save each patch into file directory
#                 cv2.imwrite(os.path.join(out_dir,'tile_'+str(x)+'_'+str(y)+'.jpg'), tile)
#                 tiles.append(tile)
#                 limits.append((x, y, x1, y1))
      
#             else:
#                 #Crop into patches of size MxN
#                 tile = image_copy[y:y+M, x:x+N]
#                 #Save each patch into file directory
#                 cv2.imwrite(os.path.join(out_dir,'tile_'+str(x)+'_'+str(y)+'.jpg'), tile)
#                 # cv2.rectangle(image, (x, y), (x1, y1), (0, 255, 0), 1)      
#                 tiles.append(tile)
#                 limits.append((x, y, x1, y1))
#         # if viz:
#         #     i = 1
#         #     for tile in tiles:
#         #         ax1 = plt.subplot(len(tiles),1,i)
#         #         ax1.imshow(tile)
#         #         plt.show()       
#         #         i += 1  
  
#     return tiles, limits


# def subdivideImage(image, x1=0, y1=0, M=1000, N=2000, viz=False, out_dir='tiles'):
#         # if image is None:
#     #     print('Invalid image input')
#     #     return None, None
#     assert not (image is None), 'Invalid image input'
    
#     isExist = os.path.exists(out_dir)
#     if not isExist:
#         os.makedirs(out_dir)

    
#     image_copy = image.copy()
#     imgheight = image.shape[0]
#     imgwidth = image.shape[1]
       
#     tiles = [] 
#     limits = []
    
#     for y in range(0, imgheight, M):
#         for x in range(0, imgwidth, N):
#             if (imgheight - y) < M or (imgwidth - x) < N:
#                 break
                
#             y1 = y + M
#             x1 = x + N
    
#             # check whether the patch width or height exceeds the image width or height
#             if x1 >= imgwidth and y1 >= imgheight:
#                 x1 = imgwidth - 1
#                 y1 = imgheight - 1
#                 #Crop into patches of size MxN
#                 tile = image_copy[y:y+M, x:x+N]
#                 #Save each patch into file directory
#                 cv2.imwrite(os.path.join(out_dir,'tile_'+str(x)+'_'+str(y)+'.jpg'), tile)
#                 tiles.append(tile)
#                 limits.append((x, y, x1, y1))
                
#             elif y1 >= imgheight: # when patch height exceeds the image height
#                 y1 = imgheight - 1
#                 #Crop into patches of size MxN
#                 tile = image_copy[y:y+M, x:x+N]
#                 #Save each patch into file directory
#                 cv2.imwrite(os.path.join(out_dir,'tile_'+str(x)+'_'+str(y)+'.jpg'), tile)
#                 tiles.append(tile)
#                 limits.append((x, y, x1, y1))

#             elif x1 >= imgwidth: # when patch width exceeds the image width
#                 x1 = imgwidth - 1
#                 #Crop into patches of size MxN
#                 tile = image_copy[y:y+M, x:x+N]
#                 #Save each patch into file directory
#                 cv2.imwrite(os.path.join(out_dir,'tile_'+str(x)+'_'+str(y)+'.jpg'), tile)
#                 tiles.append(tile)
#                 limits.append((x, y, x1, y1))
      
#             else:
#                 #Crop into patches of size MxN
#                 tile = image_copy[y:y+M, x:x+N]
#                 #Save each patch into file directory
#                 cv2.imwrite(os.path.join(out_dir,'tile_'+str(x)+'_'+str(y)+'.jpg'), tile)
#                 # cv2.rectangle(image, (x, y), (x1, y1), (0, 255, 0), 1)      
#                 tiles.append(tile)
#                 limits.append((x, y, x1, y1))
#                
#     if viz:
#         i = 1
#         for tile in tiles:
#             ax1 = plt.subplot(len(tiles),1,i)
#             ax1.imshow(tile)
#             plt.show()       
#             i += 1
  
#     return tiles, limits



# rowDivisor = 2
# colDivisor = 2
# image = cv2.imread('t0/img/IMG_0520.JPG', cv2.IMREAD_COLOR)
# 
# def subdivideImage(image, rowDivisor, colDivisor):
#     if image is None:
#         return None, None, None
    
#     # Check image dimensions
#     if not image.shape[1] % colDivisor == 0 or not image.shape[0] % rowDivisor == 0 :
#         print('warning')
#         if not image.shape[1] % colDivisor == 0:
#             print('col.. cropping image... removing last column')
#             image = image[:, 0:-1]

#         if not image.shape[0] % rowDivisor == 0:
#             print('row... cropping image... removing last row...')
#             image = image[0:-1, :]

#     if image.shape[1] % colDivisor == 0 and image.shape[0] % rowDivisor == 0:
#         for
#         y = 0; y < img.cols; y += img.cols / colDivisor)
#             blocks.push_back(img(cv2.Rect(y, x, (img.cols / colDivisor), (img.rows / rowDivisor))).clone());
#     elif not image.shape[1] % colDivisor == 0


# r = cv2.selectROI("select the area", im0)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# cropped_image = im0[int(r[1]):int(r[1]+r[3]),
#                       int(r[0]):int(r[0]+r[2])]
# # Display cropped image
# cv2.imshow("Cropped image", cropped_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# // function for splitting image into multiple blocks. rowDivisor and colDivisor specify the number of blocks in rows and cols respectively
#         int subdivide(const cv::Mat &img, const int rowDivisor, const int colDivisor, std::vector<cv::Mat> &blocks)
#         {        
#             /* Checking if the image was passed correctly */
#             if(!img.data || img.empty())
#                 std::cerr << "Problem Loading Image" << std::endl;

#             /* Cloning the image to another for visualization later, if you do not want to visualize the result just comment every line related to visualization */
#             cv::Mat maskImg = img.clone();
#             /* Checking if the clone image was cloned correctly */
#             if(!maskImg.data || maskImg.empty())
#                 std::cerr << "Problem Loading Image" << std::endl;

#             // check if divisors fit to image dimensions
#             if(img.cols % colDivisor == 0 && img.rows % rowDivisor == 0)
#             {
#                 for(int y = 0; y < img.cols; y += img.cols / colDivisor)
#                 {
#                     for(int x = 0; x < img.rows; x += img.rows / rowDivisor)
#                     {
#                         blocks.push_back(img(cv::Rect(y, x, (img.cols / colDivisor), (img.rows / rowDivisor))).clone());
#                         rectangle(maskImg, Point(y, x), Point(y + (maskImg.cols / colDivisor) - 1, x + (maskImg.rows / rowDivisor) - 1), CV_RGB(255, 0, 0), 1); // visualization

#                         imshow("Image", maskImg); // visualization
#                         waitKey(0); // visualization
#                     }
#                 }
#             }else if(img.cols % colDivisor != 0)
#             {
#                 cerr << "Error: Please use another divisor for the column split." << endl;
#                 exit(1);
#             }else if(img.rows % rowDivisor != 0)
#             {
#                 cerr << "Error: Please use another divisor for the row split." << endl;
#                 exit(1);
#             }
#         return EXIT_SUCCESS;
#     }
        
        # from image_slicer import slice
        # import cv2 
        # import numpy as np

        # a = slice('t0/img/IMG_0520.JPG',4)
        # tile = a[0].image
        # tile = tile.convert("RGB")
        # tile =  np.asarray(tile)
        # cv2.imshow('tile 0', tile)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()