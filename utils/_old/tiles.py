import numpy as np
import cv2 
import os           
# import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('qt5agg')
                
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

def generateTiles(image, rowDivisor=2, colDivisor=2, overlap = 200, viz=False, out_dir='tiles', writeTile2Disk=True):
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
            tiles.append(tile)
            if writeTile2Disk:
                isExist = os.path.exists(out_dir)
                if not isExist:
                    os.makedirs(out_dir)       
                cv2.imwrite(os.path.join(out_dir,'tile_'+str(tileIdx)+'_'
                         +str(limits[tileIdx][0])+'_'+str(limits[tileIdx][1])+'.jpg'), tile)

    return tiles, limits
       
# def appendPred(pred, predTile):
#     # keys0 = [ key for key in pred.keys() if 'descriptors' not in key]
#     predNew = {k: np.append(pred[k], v, axis=0) for k, v in predTile.items() if 'descriptors' not in k}  
#     predNew['descriptors0'] = np.append(pred['descriptors0'], predTile['descriptors0'], axis=1)
#     predNew['descriptors1'] = np.append(pred['descriptors1'], predTile['descriptors1'], axis=1)
#     return predNew   

# def applyMatchesOffset(matches, offset):
#     for i, mtch in enumerate(matches):
#         if matches[i] > -1: 
#             matches[i] = mtch + offset
            
#     return matches  