#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import copy
import matplotlib.pyplot as plt

from sklearn.neighbors import KDTree
from scipy.interpolate import UnivariateSpline

from get_cone_pos import Get_cone_pos

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
                if child1.tolist() == points[0].tolist() or child2.tolist() == points[0].tolist(): # form loop closure
                    path.append(points[0])
                break
    
    # put path together
    if path1 != []:
        total_path = path1.reverse()
        if path != []:
            total_path += path
    else:
        total_path = path
    
    return total_path


def get_path(red, blue):
    # get the red and blue cone pose
    # generate the central path
    if len(red) > len(blue):
        choice_tree = red
        choice_not_tree = blue
    else:
        choice_tree = blue
        choice_not_tree = red
    
    tree = KDTree(choice_tree)
    
    central_path_pts = []
    for pt in choice_not_tree:
        _, ind = tree.query(pt.reshape(1,-1),1) # get the nearest other color cone
        neighbor = choice_tree[ind[0][0]]
        central_path_pts.append((neighbor + pt) / 2)
    
    # sort the central path pts
    central_pt_2d = np.array(central_path_pts)[:,0:2] # only use x,y coordinate
    total_path = sort_path_pt(central_pt_2d)
    return total_path

def visualize(red, blue, total_path):
    total_path_T = np.array(total_path).T
    red = red.T
    blue = blue.T
    plt.plot(total_path_T[0],total_path_T[1],'g-', red[0],red[1],"ro",blue[0],blue[1],"bo")
    plt.show()

def main():
    get_clone = Get_cone_pos()
    get_clone.read_file()
    red, blue = get_clone.get_cones()
    #get_clone.visualize()
    red = np.array(red)
    blue = np.array(blue)
    path = get_path(red,blue) # [np.array([x,y]),np.array([x,y]),...]
    #visualize(red,blue,path)

    # path smoothing
    distance = np.cumsum( np.sqrt(np.sum( np.diff(path,axis=0)**2, axis = 1)))
    distance = np.insert(distance, 0, 0)/distance[-1]
    
    # Build a list of the spline function, one for each dimension:
    splines = [UnivariateSpline(distance, coords, k=3, s=.2) for coords in np.array(path).T]
    
    alpha = np.linspace(0,1.0, 3 * len(distance))
    smoothed_path = np.vstack( [spl(alpha) for spl in splines] ).T
    visualize(red,blue,smoothed_path)
    return smoothed_path

if __name__ == "__main__":
    smoothed_path = main()
    np.savetxt('smoothed_fake_central_path.txt',smoothed_path, fmt = '%f')
        