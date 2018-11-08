
# tools for semantic segmentation

import numpy as np

AREA_THRESHOLD = 500
#################################################
# seg at 165, caffe
class seg_at_165():
    def __init__(self):
        self.labels = {'sky':0, 'building':1, 'column_pole':2, 'lane':3 ,'road':4, '':5, 'tree':6, 'roadshoulder':7, 'Fence':8, 'car':9, 'pedestrian':10, 'bicyclist':11}

        self.colorlabel = cv2.imread('./segphoto/camvid12.png')

    # show in color
    def color_ids(self, imgseg):
        '''
        img3ch = np.resize(imgseg, (3,360,480)).astype(np.uint8)
        img3ch = img3ch.transpose(1,2,0)
        rgb = np.zeros(img3ch.shape, dtype=np.uint8)
        print(imgseg.shape, img3ch.shape, rgb.shape, img3ch)
        cv2.LUT(img3ch, colorlabel, rgb)
        '''
        rgb = self.colorlabel[0][imgseg]
        
        cv2.imshow('color', rgb)
        cv2.imwrite('rgb.png',rgb)
        cv2.waitKey(1)


#################################################
# seg at 138, tf
class seg_at_138():
    def __init__(self):
        self.ID_CHANGE = {'0':255, '1':255, '2':255, '3':255, '4':255, '5':255, '6':255, '7':0, '8':1, '9':255, '10':255, '11':2, '12':3, '13':4, '14':255,'15':255,'16':255, '17':5, '18':255, '19':6, '20':7, '21':8, '22':9, '23':10, '24':11, '25':12, '26':13, '27':14, '28':15, '29':255, '30':255, '31':16, '32':17, '33':18, '-1':-1}
        self.ID_C = np.array([-1,-1,-1,-1,-1, -1,-1,0,1,-1, -1,2,3,4,-1, -1,-1,5,-1,6, 7,8,9,10,11, 12,13,14,15,-1, -1,16,17,18,-1], dtype=np.int16)
        self.colors = [[128,64,128], [244,35,232], [70,70,70], [102,102,156], [190,153,153], [153,153,153], [250,170,30], [220,220,0], [107,142,35], [152,251,152], [255,0,0], [220,20,60], [70,130,180], [0,0,142], [0,0,70], [0,60,100], [0,80,100], [0,0,230], [119,11,32], [0,0,0]]
        self.ALL_TYPES = {'0':'road', '1':'sidewalk', '2':'building', '3':'wall', '4':'fence', '5':'pole', '6':'trafficc light', '7':'traffic sign', '8':'vegetation', '9':'terrain', '10':'sky', '11':'person', '12':'rider', '13':'car', '14':'truck', '15':'bus', '16':'train', '17':'motorcycle', '18':'bike', '-1':'unknown',}
        self.labels = {'road':0, 'sidewalk':1, 'building':2, 'wall':3, 'fence':4, 'pole':5, 'traffic light':6, 'traffic sign':7, 'vegetation':8, 'terrain':9, 'sky':10, 'person':11, 'rider':12, 'car':13, 'truck':14, 'bus':15, 'train':16, 'motorcycle':17, 'bike':18, 'unknown':-1, }

    # change origin id into ALL_TYPES and show in color
    def change_the_ori_typeid(self,type_id):
        '''
        colorimg = []
        maxid = []
        #print(type_id.shape)
        #countover19 = 0
        for i in range(type_id.shape[0]):
            for j in range(type_id.shape[1]):
                if(type_id[i][j] < 34):
                    realid = ID_CHANGE[str(type_id[i][j])]
                if(realid >= 19):
                    #countover19 += 1
                    realid = -1
                colorimg.append(colors[realid])
        colorimg = np.array(colorimg,dtype='uint8')
        colorimg = colorimg.reshape((960,1280,3))
        colorimg = cv2.resize(colorimg, (640,480))
        '''
        #print(ID_C.dtype, type_id.dtype)
        idc = self.ID_C[type_id]

        return idc

    #remember to -1 when read the id from image
    def read_saved_id(self,filename):
        ids = cv2.imread(filename, cv2.IMREAD_UNCHANGED).astype(np.int16) - 1
        return ids

    def color_ids(self,ids):
        colorimg = self.colors[ids]
        cv2.imshow('color', colorimg)
        cv2.waitKey(1)

    def get_class_mask(self, segimg, classname='unknown'):
        lookupmask = np.zeros(len(self.colors), dtype=np.uint8)
        if(not segimg == type(lookupmask)):
            print('[seg_at_138]input image should be numpy.ndarray')
            return None
        if(not classname in self.labels.keys()):
            print('[seg_at_138]request class not in list')
            return None
        
        lookupmask[self.labels[classname]] = 255
        maskimg = lookupmask[segimg]
        return maskimg

    def find_objs_in_classmask(self, maskimg):
        _, maskbinary = cv2.threshold(maskimg, 128, 1, cv2.THRESH_BINARY)
        numofpixs = np.sum(maskbinary)

        kernel = np.ones((3,3),np.uint8)
        tmpimg = cv2.dilate(maskbinary, kernel, iterations = 2)
        tmpimg = cv2.erode(tmpimg, kernel, iterations = 3)
        tmpimg = cv2.dilate(tmpimg, kernel, iterations = 2)
        _, contours, _ = cv2.findContours(tmpimg, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        objs = []
        for i in range(len(contours)):
            M = cv2.moments(contours[i])
            area = M['m00']
            center = [int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])]
            if(area > AREA_THRESHOLD):
                objs.append({'id':i, 'size':area, 'center':center, 'contours':contours[i]})

        return objs
