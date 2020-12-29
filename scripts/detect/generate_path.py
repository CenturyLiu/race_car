#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import copy
import matplotlib.pyplot as plt

from sklearn.neighbors import KDTree
from scipy.interpolate import UnivariateSpline



def in_list(elt, array):
    for array_elt in array:
        #print(elt)
        #print(array_elt)
        if elt.tolist() == array_elt.tolist():
            return True
    return False

def sort_path_pt(points):
    # given a series of unordered path point, link the points together.
    path = []
    path1 = []
    #path2 = []
    
    # form a KDtree using input points
    tree = KDTree(points)
    
    pt = points[0] # choose the first point as initial point
    path.append(pt)
    
    other_side = []
    
    

    # create the path
    while True:
        # for each points, find 4 nearest neighbors (3 other points pt1,pt2,pt3, one is self)
        _, ind = tree.query(pt.reshape(1,-1), k=4)
        
        pt1 = points[ind[0][1]]
        pt2 = points[ind[0][2]]
        pt3 = points[ind[0][3]]
        # check whether the angle between pt-pt1 and pt-pt2 is large enough
        # i.e. vector dot product negative
        vec1 = np.array(pt1 - pt)
        vec2 = np.array(pt2 - pt)
        
        if np.dot(vec1, vec2) < 0:
            child1 = np.array(pt1)
            child2 = np.array(pt2)
        else:
            child1 = np.array(pt1)
            child2 = np.array(pt3)
        
        #print(child1)
        # check whether child1, child2 have been in "path"
        if in_list(child1,path) and not in_list(child2,path):#child1 in path and child2 not in path:
            path.append(child2)
            pt = child2 # add child2 to path, then expand child2
        elif not in_list(child1,path) and in_list(child2,path):#child1 not in path and child2 in path:
            path.append(child1)
            pt = child1
        elif not in_list(child1,path) and not in_list(child2,path):#child1 not in path and child2 not in path:
            path.append(child1)
            other_side = child2 # used to search in the other direction
            pt = child1
        elif in_list(child1,path) and in_list(child2,path):#child1 in path and child2 in path:
            if other_side != [] and not in_list(other_side, path):
                pt = other_side # change direction
                path1 = copy.copy(path)
                path = [] # new results in another list
                path.append(points[0])
                path.append(pt)
                other_side = []
            else:
                if (child1.tolist() == points[0].tolist() or child2.tolist() == points[0].tolist()) and np.dot(child1 - pt, child2 - pt) < 0: # form loop closure
                    path.append(points[0])
                break
    print("---")
    print('path1:',path1)
    print('path:',path)
    # put path together
    if path1 != []:
        path1.reverse()
        total_path = path1
        if path != []:
            total_path += path
    else:
        total_path = path
    
    return total_path
