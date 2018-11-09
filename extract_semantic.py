
# extract all feature of all photos

import cv2
import time
import math
import numpy as np
from PIL import Image

from semantic_tools import *
from data_list_test36in1 import *

def make_sedataset():
    seg138 = seg_at_138()
    segfilepath = '/home/biatc-admin/Record/getseg/segres36in1tf/'
    #outfile = open('semantic_datas_36in1.py','a+')
    #outfile.write('semantic_dataset=[')
    semantic_dataset = []
    for i in range(0,len(data_list_test)):
        print('-------------------{} in {}'.format(i,len(data_list_test)))
        onedescribe = {'name':data_list_test[i],'id':int(i/36),'dir':float(i%36)/18*math.pi,'objs':{},}
        segimg = seg138.read_saved_id(segfilepath+data_list_test[i].strip('jpg')+'png')
        for c in seg138.staticclass:
            maskimg = seg138.get_class_mask(segimg, c)
            objs = seg138.find_objs_in_classmask(maskimg)
            if(len(objs) > 0):
                onedescribe['objs'][c] = objs
        #outfile.write(str(onedescribe).replace(' ','')+',\n')
        semantic_dataset.append(onedescribe)
    np.save('output/semantic_dataset.npy', semantic_dataset)
    #outfile.write(']')
    #outfile.close()
    print('done')

def make_searchtree():
    #from semantic_datas_36in1 import *
    #outfile = open('search_tree.py','w')
    #outfile.write('semantic_tree={')
    semantic_dataset = np.load('output/semantic_dataset.npy')
    seg138 = seg_at_138()
    tree = seg138.build_search_tree(semantic_dataset)
    #print(semantic_dataset)
    #print(tree)
    for c in tree['pocess'].keys():
        print('class '+c+' has {} imgs'.format(len(tree['pocess'][c])))
        for charactor in tree['pocess'][c].keys():
            print('  charactor '+charactor)
            for des in tree['pocess'][c][charactor].keys():
                print('    '+des+' has {} imgs.'.format(len(tree['pocess'][c][charactor][des])))
    np.save('output/search_tree.npy', [tree])
    #outfile.write(str(tree))
    #outfile.write('}')
    #outfile.close()
    print('done')

def test_locate_img():
    seseg_cli = zmq_comm_cli_c(name='seg', ip='192.168.22.138', port=1380)
    print('get seg')
    img = Image.open('tmp.jpg')
    img=img_rgb_to_jpeg(img)
    st = time.time()
    res=seseg_cli.execute(img)
    print('change id')
    seg138 = seg_at_138()
    segimg = seg138.change_the_ori_typeid(res)
    print('load tree')
    search_dataset = np.load('output/search_tree.npy')[0]
    print('cmp')
    prob, dirs = seg138.locate_segphoto(segimg, search_dataset)
    print(prob,dirs)

if __name__ == '__main__':
    #make_sedataset()
    #make_searchtree()
    test_locate_img()
