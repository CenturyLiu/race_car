#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 14:27:07 2020

@author: centuryliu
"""

import numpy as np
from sklearn.neighbors import KDTree
import math
from scipy.interpolate import UnivariateSpline
import rospkg
from configobj import ConfigObj


class PurePursuit(object):
    def __init__(self, speed = 3.5, Lfc = 4.0):
        # get directory to param file
        self.rospack = rospkg.RosPack()
        self.param_file = self.rospack.get_path('auto_pilot_param') + '/params/racecar_params' 
        
        # get params
        self.config = ConfigObj(self.param_file)
        
        
        # pure-pursuit model constants
        self.k = float(self.config['k'])#0.1 # coefficient for look ahead
        self.Lfc = Lfc # look ahead distance
        self.L = float(self.config['L'])#1.6 # distance between front and rear tires
        
        self.speed = speed
        self.Lf = self.k * self.speed + self.Lfc

    def get_speed_steer(self,path, current_xy, current_yaw):
        '''
        
    
        Parameters
        ----------
        path : list of [x,y]
            path for the car to follow
        current_xy : [x,y]
            current position of the car, should be in the same coordinate system as the path
        current_yaw : float
            yaw angle, representing the direction of the car
    
        Returns
        -------
        control : list
            [speed, steer_angle]
    
        '''
        # path smoothing
        distance = np.cumsum( np.sqrt(np.sum( np.diff(path,axis=0)**2, axis = 1)))
        distance = np.insert(distance, 0, 0)/distance[-1]
    
        # Build a list of the spline function, one for each dimension:
        splines = [UnivariateSpline(distance, coords, k=3, s=.2) for coords in np.array(path).T]
    
        alpha = np.linspace(0,1.0, 20 * len(distance))
        smoothed_path = np.vstack( [spl(alpha) for spl in splines] ).T
        
        
        path_list = smoothed_path
        path_tree = KDTree(path_list)
        
        direction_vector = np.array([np.cos(current_yaw),np.sin(current_yaw)])
        pos_2d = np.array(current_xy)
        
        # search target
        # find the points within Lf
        ind, distance = path_tree.query_radius(pos_2d.reshape(1,-1),self.Lf,return_distance = True)
        
        # sort those points based on distance
        distance_order = distance[0].argsort()
        
        target = [] # empty target    
        
        for ii in range(len(distance_order)-1,-1,-1):
            temp_target_ind = ind[0][distance_order[ii]]
            temp_pos_2d = path_list[temp_target_ind]
            # check whether the points is in front of the car
            cos_alpha = np.dot(temp_pos_2d - pos_2d, direction_vector)
            if cos_alpha > 0:
                target = temp_pos_2d
                break
        
        if target == []:
            print("No feasible target found! Stop car")
            return [0.0, 0.0]
        else:
            # calculate between current direction and target direction
            alpha = math.atan2(target[1] - pos_2d[1],target[0] - pos_2d[0]) - current_yaw
            # calculate steer angle
            delta = math.atan2(2.0 * self.L * math.sin(alpha) / self.Lf, 1.0)
            return [self.speed, delta]
            
