
# extract all feature of all photos

import cv2
import time
import math
import numpy as np
from PIL import Image

from semantic_tools import *
from data_list_test36in1 import *

allclass = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky']
if __name__ == '__main__':
    seg138 = seg_at_138()
    segfilepath = '/home/biatc-admin/Record/getseg/segres/'
    outfile = open('semantic_datas_36in1.py','a+')
    outfile.write('semantic_dataset=[')
    for i in range(len(data_list_test)):
        print('-------------------{} in {}'.format(i,len(data_list_test)))
        onedescribe = {'name':data_list_test[i],'id':int(i/36),'dir':float(i%36)/18*math.pi,'objs':{},}
        segimg = seg138.read_saved_id(segfilepath+data_list_test[i].strip('jpg')+'png')
        print(type(segimg))
        for c in allclass:
            maskimg = seg138.get_class_mask(segimg, c)
            objs = seg138.find_objs_in_classmask(maskimg)
            if(len(objs) > 0):
                onedescribe['objs'][c] = objs
        outfile.write(str(onedescribe).replace(' ','')+',\n')

    outfile.write(']')
    outfile.close()
