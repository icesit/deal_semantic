
# try to do as 'hybrid relocation in large scale MTS map'

import os
import math
import cv2
import numpy as np
import time
import random
import pickle
#from numba import jit

#from local_semantic_graph import *

ROW = 384
COL = 384
IMG_SZ = [ROW, COL]
TOTALPIXELS = ROW * COL
# wei xi tong
'''
ALL_TYPES = {'2':'building', '3':'sky', '5':'tree', '7':'road', '10':'grass', '12':'pavement', '18':'flowerbed', '21':'car', '44':'guideboard', 'other':'unknow'}
TYPE_RANK = {'2':0, '3':1, '5':2, '7':3, '10':4, '12':5, '18':6, '21':7, '44':8, 'other':9}
CLASS_CMP_WEIGHTS_DIC = {'2':2, '3':0.3, '5':0.3, '7':0.3, '10':0.5, '12':1, '18':1, '21':0, '44':2.5, 'other':0.5}
CLASS_CMP_WEIGHTS = [2, 0.3, 0.3, 0.3, 0.5, 1, 1, 0, 2.5, 0.5]
'''
# frrn
ID_CHANGE = {'0':255, '1':255, '2':255, '3':255, '4':255, '5':255, '6':255, '7':0, '8':1, '9':255, '10':255, '11':2, '12':3, '13':4, '14':255,'15':255,'16':255, '17':5, '18':255, '19':6, '20':7, '21':8, '22':9, '23':10, '24':11, '25':12, '26':13, '27':14, '28':15, '29':255, '30':255, '31':16, '32':17, '33':18, '-1':-1}
ALL_TYPES = {'0':'road', '1':'sidewalk', '2':'building', '3':'wall', '4':'fence', '5':'pole', '6':'trafficc light', '7':'traffic sign', '8':'vegetation', '9':'terrain', '10':'sky', '11':'person', '12':'rider', '13':'car', 'other':'unknow'}
TYPE_RANK = {'0':0, '1':1, '2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9, '10':10, '11':11, '12':12, '13':13, 'other':14}
CLASS_CMP_WEIGHTS_DIC = {'0':0.2, '1':1, '2':2, '3':1, '4':1, '5':1, '6':2, '7':2, '8':0.5, '9':0.5, '10':0.2, '11':0, '12':0, '13':0, 'other':0.5}
CLASS_CMP_WEIGHTS = [0.2, 1, 2, 1, 1, 1, 2, 2, 0.5, 0.5, 0.2, 0, 0, 0, 0.5]


#CONTENT_CMP_WEIGHTS = {'size':1, 'conlen':1, 'shape':1, 'links':1}
CONTENT_CMP_WEIGHTS = [1, 1, 1, 1]
AREA_THRESHOLD = 300
# IMG_FILE_PATH = '../resultseg'
# CSV_PATH_HEAD = '../csv/seg_'
# PKL_FILE_PATH = 'smalldatabasetest/'

IMG_FILE_PATH = '../162621seg'
PKL_FILE_PATH = '162621seg/'
OUTPUT_FILE_PATH = PKL_FILE_PATH#'1seg/'

def gen_fname_list(rootdir='./'):
    flist=[]
    for pname,dnames,fnames in os.walk(rootdir):      
        flist +=[ os.path.join(pname,fname) for fname in fnames]
    flist.sort()
    return flist

# @ input:
#   o1, o2: {'id':, 'size':, 'center':, 'contours':}
# @ return:
#   link_percent: num_of_pixels/pixels_of_self_contour
def find_link_between_two_obj(o1, o2):
    img1 = np.zeros(IMG_SZ, dtype='uint8')
    img2 = np.zeros(IMG_SZ, dtype='uint8')
    link = 0
    cv2.drawContours(img1, [o1['contours']], -1, (120,120,120), cv2.FILLED)
    cv2.drawContours(img2, [o2['contours']], -1, (120,120,120), cv2.FILLED)
    img = img1 + img2
    # the overlap is link area
    ret, binary = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
    '''
    cv2.imshow('1',img1)
    cv2.imshow('2',img2)
    cv2.imshow('3',img)
    cv2.imshow('4',binary)
    cv2.waitKey(0)
    '''
    link = np.sum(binary) / 255
    return link

# @ input:
#   objs: [{'id':, 'size':, 'center':, 'contours':...}, obj2, ...]
# after the function, 'links':num_of_pixels/pixels_of_self_contour
def find_links_between_objs(objs):
    for i in range(len(objs)):
        for j in range(i, len(objs)):
            if(objs[i]['type'] == objs[j]['type']):
                continue
            link = find_link_between_two_obj(objs[i], objs[j])
            if(link > 0):
                objs[i]['links'][str(objs[j]['id'])] = link
                objs[j]['links'][str(objs[i]['id'])] = link
        #del objs[i]['contours']

# @ input:
#   g1, g2: [obj1,obj2...]
# @ return:
#   score: [0,1],simularity, higher = more simular
def compare_two_graph(g1, g2):
    score = 0
    # normalize
    #z = 
    leng1 = len(g1)
    leng2 = len(g2)
    if(leng1 > leng2):
        lg = g1
        sg = g2
        lenl = leng1
        lens = leng2
    else:
        lg = g2
        sg = g1
        lenl = leng2
        lens = leng1
    #print(len(g1), len(g2))
    weights = []#np.zeros(lengl, dtype='float32')
    allscore = []#np.zeros(lengl, dtype='float32')
    usedkeys_in_lg = []
    no_matched_nodes = []
    i = 0
    for k1 in range(lens):
        #tmpweights = []
        n1 = sg[k1]
        if(n1['type'] == 'other'):
            continue
        tmpallscore = []
        tmpkeys = []
        for k2 in range(lenl):
            n2 = lg[k2]
            if(n1['type'] == n2['type']):
                sco,deltas = cmp_nodes(n1, n2, sg, lg)
                #tmpweights.append(CLASS_CMP_WEIGHTS_DIC[n1['type']])
                tmpallscore.append(sco)
                tmpkeys.append(k2)
        if(len(tmpallscore)>0):
            simularone = np.argmax(tmpallscore)
            allscore.append(tmpallscore[simularone])
            usedkeys_in_lg.append(tmpkeys[simularone])
            weights.append(CLASS_CMP_WEIGHTS_DIC[n1['type']])
        elif(not n2['type'] == 'other'):
            allscore.append(0)
            weights.append(CLASS_CMP_WEIGHTS_DIC[n1['type']])
        #else:
            # no match object in larger graph
        #    no_matched_nodes.append(n1)
        i += 1
    #deal with no matched nodes
    '''
    for i in range(lenl):
        if(not i in usedkeys_in_lg):
            no_matched_nodes.append(lg[i])
    for ns in no_matched_nodes:
        sco,_ = cmp_nodes(ns)
    '''

    weights = np.array(weights)
    #print(len(weights), len(allscore))
    score = weights*allscore / np.sum(weights)
    #print(score)
    score = np.sum(score)
    return score

# @ input:
#   n1, n2: {'id':, 'size':, 'center':, 'links':}, nodes of same class from two frame
#   g1, g2: 
# @ output:
#   score: matched score of nodes
def cmp_nodes(n1, n2=None, g1=None, g2=None):
    if(n1 == None):
        print('[cmp_nodes]no nodes are compared')
        return None,None
    elif(n2 == None):
        #only one node is input
        alldels = [-1,-1,-1,-1]
        score = CONTENT_CMP_WEIGHTS * np.fabs(alldels)
        score = np.sum(score) / np.sum(CONTENT_CMP_WEIGHTS)
        score = math.exp( - score/n1['size'])
    else:
        cen1 = n1['center']
        cen2 = n2['center']
        d_center = (cen1[0]-cen2[0])**2 + (cen1[1]-cen2[1])**2
        if(d_center < 2500): # 50pixel
            delsize = (n2['size'] - n1['size']) / (n2['size'] + n1['size'])
            delconlen = (n2['conlen'] - n1['conlen']) / (n2['conlen'] + n1['conlen'])
            delcenter = [cen2[1]-cen1[1], cen2[0]-cen1[0]]
            linkvec1 = np.ones(len(ALL_TYPES),dtype='float32') / 10000
            linkvec2 = np.ones(len(ALL_TYPES),dtype='float32') / 10000
            for lk1 in n1['links'].keys():
                linkvec1[TYPE_RANK[g1[int(lk1)]['type']]] += n1['links'][lk1]
            for lk2 in n2['links'].keys():
                linkvec2[TYPE_RANK[g2[int(lk2)]['type']]] += n2['links'][lk2]
            dellinkvec = (linkvec2-linkvec1)/(linkvec2+linkvec1)
            dellink = CLASS_CMP_WEIGHTS * dellinkvec
            dellink = np.sum(dellink) / np.sum(CLASS_CMP_WEIGHTS)
            delshape = cv2.matchShapes(n1['contours'], n2['contours'], 1, 0.0)

            alldels = [delsize, delconlen, delshape, dellink]
            score = CONTENT_CMP_WEIGHTS * np.fabs(alldels)
            score = np.sum(score) / np.sum(CONTENT_CMP_WEIGHTS)
            score = math.exp( - score)
            #print(n1['type'],n2['type'],alldels,score)
            #(delsize*CONTENT_CMP_WEIGHTS['size'] + delconlen*CONTENT_CMP_WEIGHTS['conlen'] + delshape*CONTENT_CMP_WEIGHTS['shape'] + dellink*)
        else:
            alldels = []
            score = 0

    return score, alldels

# @ input:
#   img_this_class: mask of one class
#   imgseg: a semantic result image
#   cla: type of class
# @ return:
#   objs: all objects list
def extract_semantic_feature_of_one_class(img_this_class, imgseg, cla):
    #find contours
    tmpimg = img_this_class
    
    kernel = np.ones((3,3),np.uint8)
    tmpimg = cv2.dilate(img_this_class, kernel, iterations = 2)
    tmpimg = cv2.erode(tmpimg, kernel, iterations = 3)
    tmpimg = cv2.dilate(tmpimg, kernel, iterations = 2)
    '''
    '''
    _, contours, _ = cv2.findContours(tmpimg, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    #keep contours thich are large enough
    objs = []
    for i in range(len(contours)):
        M = cv2.moments(contours[i])
        area = M['m00']#cv2.contourArea(contours[i])
        center = [int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])]
        contourslen = cv2.arcLength(contours[i], True)
        #print(M)
        #print(area, contourslen)
        if(area > AREA_THRESHOLD):
            #cal links
            objs.append({'id':i, 'size':area, 'center':center, 'type':cla, 'conlen':contourslen, 'links':{}, 'contours':contours[i], 'M':M})
            #, 'contours':contours[i]
    return objs

# a local semantic graph includes every type object in the image
# @ input:
#   imgseg: a semantic result image, every pixel is the class number in ALL_TYPES, the other number will all be other type
# @ return:
#   local_semantic_graph:
def build_local_semantic_graph(imgseg):
    p1time = time.time()
    # make every class an image of 0 and 1
    alltype_imgs = {'other':np.zeros_like(imgseg, dtype='uint8')}
    #objects_of_each_class = {'other':[]}
    objs = []
    for k in ALL_TYPES.keys():
        alltype_imgs[k] = np.zeros_like(imgseg, dtype='uint8')
        #objects_of_each_class[k] = []

    for y in range(imgseg.shape[0]):
        for x in range(imgseg.shape[1]):
            oriid = str(imgseg[y,x])
            str_typeid = str(ID_CHANGE[oriid])
            # str_typeid = str(imgseg[y,x])
            if(str_typeid in ALL_TYPES.keys()):
                alltype_imgs[str_typeid][y,x] = 255
            else:
                alltype_imgs['other'][y,x] = 255

    p2time = time.time()
    # create posible objects of class and
    # add properties(position in image, num of pixels, links between objects) to object
    local_semantic_graph = {}
    i = 0
    for k in alltype_imgs.keys():
        #one_object = {'size':0, 'links':{}, 'position':[]}
        objs_of_one_class = extract_semantic_feature_of_one_class(alltype_imgs[k], imgseg, k)
        #print(objs_of_one_class)
        #objects_of_each_class[k] = objs_of_one_class
        for oneobj in objs_of_one_class:
            oneobj['id'] = i
            i += 1
            objs.append(oneobj)
    
    find_links_between_objs(objs)
    #print(objects_of_each_class)
    endtime = time.time()
    print('p1 took {} seconds\np2 took {} seconds'.format(p2time-p1time, endtime-p2time))
    return objs #objects_of_each_class

def print_local_semantic_graph(lsg):
    print('ok')
    pass

def draw_in_seg(imgname, objs):
    img = cv2.imread('resultseg/'+imgname+'.jpeg').astype(np.uint8)
    for obj in objs:
        cv2.circle(img, (obj['center'][0], obj['center'][1]), 5, (130,200,50), 2)
        cv2.drawContours(img,[obj['contours']],0,(0,0,255),1)  
    cv2.imshow('111',img)
    cv2.waitKey(1)

def test_build_local_graph():
    start = time.time()
    imgname = 'photo_lng121.442293_lat31.032693_image0'
    fname_csv = CSV_PATH_HEAD+imgname+'.csv'
    type_id = np.genfromtxt(fname_csv, delimiter=',').astype(int).reshape(ROW,COL)

    output = build_local_semantic_graph(type_id)
    print_local_semantic_graph(output)
    draw_in_seg(imgname, output)

    totaltime = time.time() - start
    print('total time is {} seconds'.format(totaltime))

def build_all_local_graph():
    start = time.time()
    flist=gen_fname_list(IMG_FILE_PATH)
    outfile = open(OUTPUT_FILE_PATH+'local_semantic_graph.pkl', 'wb')
    #outfile.write('local_graphs={\n')
    i=0
    towrite = {}
    for filename in flist:
        # read seg from csv
        '''
        filename = filename.strip('.jpeg')
        filename = filename.split('/')[-1]
        fname_csv = CSV_PATH_HEAD+filename+'.csv'
        print('image {}/{}:{}'.format(i, len(flist), filename))
        type_id = np.genfromtxt(fname_csv, delimiter=',').astype(int).reshape(ROW,COL)
        '''
        # read seg from png
        filename = filename.strip('.png')
        filename = filename.split('/')[-1]
        print(filename)
        type_id = cv2.imread(IMG_FILE_PATH+'/'+filename+'.png', cv2.IMREAD_UNCHANGED)
        
        output = build_local_semantic_graph(type_id)
        print('{} objects are in the img'.format(len(output)))
        #draw_in_seg(filename, output)
        #outfile.write('\''+filename+'\':'+str(output).replace('\n', '').replace(' ', '')+',\n')
        towrite[filename] = output
        
        #outfile.write('\n')
        i+=1
        #if(i>2):
        #    break
    #outfile.write('}')
    pickle.dump(towrite, outfile, 1)
    outfile.close()
    totaltime = time.time() - start
    print('total time is {} seconds'.format(totaltime))

def test_cmp_two_graph():
    #g1 = local_graphs['photo_lng121.445384_lat31.029858_image3']
    #g2 = local_graphs['photo_lng121.444242_lat31.030467_image1']

    pklfile = open(OUTPUT_FILE_PATH+'local_semantic_graph.pkl', 'rb')
    local_graphs = pickle.load(pklfile)
    '''
    #cmp two graph
    name1 = 'photo_lng121.444325_lat31.031795_heading69.35_northdir200.65_image_02'
    name2 = 'photo_lng121.444325_lat31.031795_heading69.35_northdir200.65_image_15'
    name3 = 'photo_lng121.444325_lat31.031795_heading69.35_northdir200.65_image_14'
    g1 = local_graphs[name1]
    g2 = local_graphs[name2]
    g3 = local_graphs[name3]
    #print(type(g1))
    s1 = compare_two_graph(g1, g2)
    print('..........')
    s2 = compare_two_graph(g1, g3)

    print('simularity between {} and {} is {}|{}'.format(name1, name2, s1, s2))
    '''
    #cmp all graph
    flist=gen_fname_list(IMG_FILE_PATH)
    outfile = open(OUTPUT_FILE_PATH+'simularitymatrix.pkl', 'wb')
    i = 0
    j = 0
    simularmatrix = np.zeros([len(flist), len(flist)])
    totallen = len(flist)
    start = time.time()
    for filename in flist:
        # wei xi tong
        '''
        filename = filename.strip('.jpeg')
        filename = filename.split('/')[-1]
        '''
        # 162621
        filename = filename.strip('.png')
        filename = filename.split('/')[-1]
        g1 = local_graphs[filename]
        j = 0
        for filename2 in flist:
            '''
            filename2 = filename2.strip('.jpeg')
            filename2 = filename2.split('/')[-1]
            '''
            filename2 = filename2.strip('.png')
            filename2 = filename2.split('/')[-1]
            g2 = local_graphs[filename2]

            s = compare_two_graph(g1, g2)
            simularmatrix[i][j] = s

            print('({},{})/({},{})simularity between {} and {} is {}'.format(i,j, totallen,totallen, filename, filename2, s))
            j += 1
        i += 1
    totaltime = time.time() - start
    print('total time is {} seconds, average {} seconds for one compare, {} seconds for one set compare'.format(totaltime, totaltime/totallen/totallen, totaltime/totallen))
    
    cv2.imshow('res', simularmatrix)
    cv2.waitKey(1)
    #outfile.write(simularmatrix)
    pickle.dump(simularmatrix, outfile, 1)
    outfile.close()
    

def test_cmp_modify_graph():
    DIFF_PERCENT = 0.1
    drow = int(ROW*(1-DIFF_PERCENT))
    dcol = int(COL*(1-DIFF_PERCENT))
    pklfile = open('local_semantic_graph0911.pkl', 'rb')
    local_graphs = pickle.load(pklfile)
    flist=gen_fname_list(IMG_FILE_PATH)
    outfile = open('modsimularitymatrix.pkl', 'wb')
    i = 0
    j = 0
    simularmatrix = np.zeros([len(flist), len(flist)])
    totallen = len(flist)
    start = time.time()
    for filename in flist:
        filename = filename.strip('.jpeg')
        filename = filename.split('/')[-1]
        fname_csv = CSV_PATH_HEAD+filename+'.csv'
        type_id = np.genfromtxt(fname_csv, delimiter=',').astype(int).reshape(ROW,COL)
        startrow = int(random.random()*DIFF_PERCENT*ROW)
        startcol = int(random.random()*DIFF_PERCENT*COL)
        type_id = type_id[startrow:(startrow+drow),startcol:(startcol+dcol)]
        tst = time.time()
        g1 = build_local_semantic_graph(type_id)
        gent = time.time() - tst

        j = 0
        for filename2 in flist:
            filename2 = filename2.strip('.jpeg')
            filename2 = filename2.split('/')[-1]
            g2 = local_graphs[filename2]

            s = compare_two_graph(g1, g2)
            simularmatrix[i][j] = s
            if(filename2==filename):
                print('({},{})/({},{})simularity between {} and {} is {} with shift({},{}), generating a graph use {} second'.format(i,j, totallen,totallen, filename, filename2, s, startrow, startcol, gent))
            j += 1
        i += 1

    totaltime = time.time() - start
    print('total time is {} seconds, average {} seconds for one compare'.format(totaltime, totaltime/totallen/totallen))
    
    #outfile.write(simularmatrix)
    pickle.dump(simularmatrix, outfile, 1)
    outfile.close()
    cv2.imshow('res', simularmatrix)
    cv2.waitKey(0)

def show_neibor_graph_cmp():
    readmatrixpkl = 'simularitymatrix'
    pklfile = open(PKL_FILE_PATH+readmatrixpkl+'.pkl', 'rb')
    simularmatrix = pickle.load(pklfile)

    for i in range(len(simularmatrix)):
        print('the {} img cmp'.format(i))
        print(simularmatrix[i])
        toshow = np.zeros(7)
        for k in range(7):
            idd = i+k-3
            if(idd >= len(simularmatrix)):
                idd -= len(simularmatrix)
            toshow[k] = simularmatrix[i][idd]
        print(toshow)

def read_and_show_simularmatrix():
    readmatrixpkl = 'simularitymatrix'
    pklfile = open(PKL_FILE_PATH+readmatrixpkl+'.pkl', 'rb')
    simularmatrix = pickle.load(pklfile)
    allaver = []
    for i in range(len(simularmatrix)):
        aver = (np.sum(simularmatrix[i])-1)/(len(simularmatrix)-1)
        allaver.append(aver)
        #print('image {} average simular score is {}'.format(i,aver))
    totalaver = np.mean(allaver)
    maxaver = np.max(allaver)
    maxaverid = np.argmax(allaver)
    minaver = np.min(allaver)
    minaverid = np.argmin(allaver)
    print('total average simular score is {}, max simular is {} of node {}, min simular is {} of node {}'.format(totalaver, maxaver, maxaverid, minaver, minaverid))
    cv2.imshow('res', simularmatrix)
    cv2.imwrite(OUTPUT_FILE_PATH+readmatrixpkl+'.jpg', simularmatrix*255)
    cv2.waitKey(1)
    pklfile.close()

if __name__ == "__main__": 
    #test_build_local_graph()
    build_all_local_graph()
    test_cmp_two_graph()
    read_and_show_simularmatrix()
    #test_cmp_modify_graph()
    #show_neibor_graph_cmp()