
import os
import numpy as np
import cv2

ID_CHANGE = {'0':255, '1':255, '2':255, '3':255, '4':255, '5':255, '6':255, '7':0, '8':1, '9':255, '10':255, '11':2, '12':3, '13':4, '14':255,'15':255,'16':255, '17':5, '18':255, '19':6, '20':7, '21':8, '22':9, '23':10, '24':11, '25':12, '26':13, '27':14, '28':15, '29':255, '30':255, '31':16, '32':17, '33':18, '-1':-1}
colors = [[128,64,128], [244,35,232], [70,70,70], [102,102,156], [190,153,153], [153,153,153], [250,170,30], [220,220,0], [107,142,35], [152,251,152], [70,130,180], [220,20,60], [255,0,0], [0,0,142], [0,0,70], [0,60,100], [0,80,100], [0,0,230], [119,11,32], [0,0,0]]
#colors = [(128,64,128), (244,35,232), (70,70,70), (102,102,156), (190,153,153), (153,153,153), (250,170,30), (220,220,0), (107,142,35), (152,251,152), (70,130,180), (220,20,60), (255,0,0), (0,0,142), (0,0,70), [0,60,100], (0,80,100), (0,0,230), (119,11,32), (0,0,0)]
colors = np.array(colors)

def gen_fname_list(rootdir='./'):
    flist=[]
    for pname,dnames,fnames in os.walk(rootdir):      
        flist +=[ os.path.join(pname,fname) for fname in fnames]
    flist.sort()
    return flist

if __name__ == '__main__':
    filepath = '../162621seg/'
    flist=gen_fname_list(filepath)

    for filename in flist:
        segname = filename.split('/')[-1] #'photo_lng121.444325_lat31.031795_heading69.35_northdir200.65_image_02.png'
        type_id = cv2.imread(filepath+segname, cv2.IMREAD_UNCHANGED)
        colorimg = []#colors[type_id]#np.zeros_like(type_id)#
        #print(type_id.shape)
        maxid = []
        countover19 = 0
        for i in range(type_id.shape[0]):
            for j in range(type_id.shape[1]):
                realid = ID_CHANGE[str(type_id[i][j])]
                if(realid >= 19):
                    countover19 += 1
                    realid = 19
                colorimg.append(colors[realid])
        colorimg = np.array(colorimg,dtype='uint8')
        colorimg = colorimg.reshape((256,512,3))
        colorimg = cv2.resize(colorimg, (512,256))
        cv2.imwrite('./segphoto/2seg/'+filename, colorimg)
        cv2.imshow('c', colorimg)
        cv2.waitKey(100)