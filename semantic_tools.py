
# tools for semantic segmentation

import numpy as np
import cv2

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
        self.staticclass = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky']
        self.wantedkey = ['size', 'center']


    # change origin id into ALL_TYPES
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
        segimg = self.ID_C[type_id]

        return segimg

    #remember to -1 when read the id from image
    def read_saved_id(self,filename):
        print(filename)
        segimg = cv2.imread(filename, cv2.IMREAD_UNCHANGED).astype(np.int16) - 1
        return segimg

    def color_segimg(self,segimg):
        colorimg = self.colors[segimg]
        cv2.imshow('color', colorimg)
        cv2.waitKey(1)

    #input the modified segimg
    def get_class_mask(self, segimg, classname='unknown'):
        lookupmask = np.zeros(len(self.colors), dtype=np.uint8)
        if(not type(segimg) == type(lookupmask)):
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
        num = 0
        for i in range(len(contours)):
            eps = 0.1*cv2.arcLength(contours[i], True)
            approx = cv2.approxPolyDP(contours[i], eps, True)
            M = cv2.moments(approx)
            area = M['m00']
            if(area > AREA_THRESHOLD):
                center = [int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])]
                objs.append({'id':num, 'size':area, 'center':center, 'contours':approx})
                num += 1

        return objs

    # search_dataset['pocess'][class]['size/center']['small/large/left/right...'] = [1,2,3,4...]
    def build_search_tree(self, sedatasets):
        # judge whether something is in the image
        pocess_dict = {}
        for c in self.staticclass:
            pocess_dict[c] = {'size':{'verylarge':[],'large':[],'middle':[],'small':[],'verysmall':[]},'center':{'left':[],'right':[],'near':[],'middle':[],'far':[]}}

        i = 0
        for onepic in sedatasets:
            for c in self.staticclass:
                if(c in onepic['objs']):
                    for obj in onepic['objs'][c]:
                        maxarea = obj['size']
                        if(maxarea < 11000):
                            pocess_dict[c]['size']['verysmall'].append(i)
                        if(9000 < maxarea < 22000):
                            pocess_dict[c]['size']['small'].append(i)
                        if(18000 < maxarea < 32000):
                            pocess_dict[c]['size']['middle'].append(i)
                        if(28000 < maxarea < 42000):
                            pocess_dict[c]['size']['large'].append(i)
                        if(38000 < maxarea):
                            pocess_dict[c]['size']['verylarge'].append(i)
                        #put in direction
                        if(obj['center'][0] < 320):
                            pocess_dict[c]['center']['left'].append(i)
                        else:
                            pocess_dict[c]['center']['right'].append(i)
                        if(obj['center'][1] < 240):
                            pocess_dict[c]['center']['far'].append(i)
                        elif(obj['center'][1] > 360):
                            pocess_dict[c]['center']['near'].append(i)
                        else:
                            pocess_dict[c]['center']['middle'].append(i)
            i += 1

        search_dataset = {'pocess':pocess_dict}
        return search_dataset

    def get_description(self, obj, wantedkey):
        res = []
        if(wantedkey == 'center'):
            if(obj['center'][0] < 320):
                res.append('left')
            else:
                res.append('right')
            if(obj['center'][1] < 240):
                res.append('far')
            elif(obj['center'][1] > 360):
                res.append('near')
            else:
                res[wantedkey].append('middle')
        elif(wantedkey == 'size'):
            maxarea = obj['size']
            if(maxarea < 11000):
                res.append('verysmall')
            if(9000 < maxarea < 22000):
                res.append('small')
            if(18000 < maxarea < 32000):
                res.append('middle')
            if(28000 < maxarea < 42000):
                res.append('large')
            if(38000 < maxarea):
                res.append('verylarge')

        return res

    #input the modified segimg
    #return prob of all places
    def locate_segphoto(self, imgseg, search_dataset):
        onedescribe = {'objs':{}}
        # get all objs of static class
        # cal prob of all imgs
        prob = np.ones(62388,dtype=np.float64) / 62388
        for c in self.staticclass:
            maskimg = self.get_class_mask(segimg, c)
            objs = self.find_objs_in_classmask(maskimg)
            if(len(objs) > 0):
                onedescribe['objs'][c] = objs
                for obj in objs:
                    for kk in self.wantedkey:
                        descri = self.get_description(obj, c, kk)
                        for de in descri:
                            prob[search_dataset['pocess'][c][kk][de]] *= 1.5
        prob.shape = (1733,36)
        prob_point = np.sum(prob, 1)
        prob_point = prob_point / np.sum(prob_point)
        dirs = np.argmax(prob, 1).astype(np.float32)/18*math.pi
        return prob_point, dirs