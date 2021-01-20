#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 19:21:38 2020

@author: centuryliu
"""
import numpy as np
import copy
import matplotlib.pyplot as plt

def in_list(elt, array):
    for array_elt in array:
        #print(elt)
        #print(array_elt)
        if elt.tolist() == array_elt.tolist():
            return True
    return False


def link_path_pt(points):
    # link a set of points
    # assume len(points) >= 3
    # points : list of np.array([x,y])
    
    # calculate distance between points
    num_pts = len(points)
    distance_mat = np.zeros((num_pts,num_pts))
    for ii in range(num_pts):
        for jj in range(1,num_pts):
            d = np.linalg.norm(points[ii] - points[jj])
            distance_mat[ii][jj] = d
            distance_mat[jj][ii] = d
    
    # form path
    path = []
    pt = points[0]
    child1 = []
    child2 = []
    id1 = None
    id2 = None
    
    ordered_index = np.argsort(distance_mat[0])
    child1 = points[ordered_index[1]] # the closest point to the initial one
    id1 = ordered_index[1]
    path.append(pt)
    path.append(child1)
    
    # check whether the initial point is end point
    vec1 = child1 - pt
    for ii in range(2,num_pts):
        temp_pt = points[ordered_index[ii]]
        if np.dot(vec1, temp_pt - pt) < 0:
            child2 = temp_pt # needs to go through the other direction
            id2 = ordered_index[ii]
            break
    
    # loop to form path
    while True:
        parent = pt
        pt = path[-1]
        ordered_index = np.argsort(distance_mat[id1])
        child1 = []
        id1 = None
        vec1 = parent - pt
        for ii in range(2,num_pts):
            temp_pt = points[ordered_index[ii]]
            if np.dot(vec1, temp_pt - pt) < 0 and not in_list(temp_pt, path):
                id1 = ii
                child1 = temp_pt
                break
        
        if len(child1) != 0:
            # found new child
            path.append(child1)
        else:
            # cannot find child => reach end point
            # check whether need to go in the other direction
            if len(child2) == 0:
                break 
            else:
                # reverse the order of the path, init point being last point
                path.reverse()
                # add child2 to path
                path.append(child2)
                id1 = id2
                # clear child2
                child2 = []
                id2 = None
    
    return path

if __name__ == "__main__":
    angles = np.pi / 6 * np.arange(7)
    np.random.shuffle(angles)
    pts = []
    for angle in angles:
        pts.append(np.array([np.cos(angle),np.sin(angle)]))
    
    path = link_path_pt(pts)
    x = []
    y = []
    for pt in path:
        x.append(pt[0])
        y.append(pt[1])
    
    plt.plot(x,y)
    
