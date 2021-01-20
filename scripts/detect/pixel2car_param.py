#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 13:49:37 2020

@author: centuryliu
"""
from configobj import ConfigObj


class Pixel2Car(object):
    def __init__(self):
        # Class for deriving cone coordinate numbers
        # in terms of car coordinate (direction of car navigation being x)
        # from the cone's center's location in terms of camera coordinate
        
        # get parameters
        
        self.config = ConfigObj('../../auto_pilot_param/params/racecar_params')
        
        # define tf constants in terms of camera frame
        self.cone_height = float(self.config['cone_height'])#0.3029 #unit: m, height of cone
        self.camera_height = float(self.config['camera_height'])#0.77 # unit: m, height of camera with respect to ground
        self.Y = self.camera_height - self.cone_height / 2 
        
        # define tf constants in terms of car frame
        self.rear_x_offset = float(self.config['rear_x_offset'])#0.775 + 0.1
        self.rear_y_offset = float(self.config['rear_y_offset'])#0.06
        
        # camera inner params
        self.fx = float(self.config['fx'])#448.13386274345095 
        self.fy = float(self.config['fy'])#448.13386274345095
        self.cx = float(self.config['cx'])#640.5
        self.cy = float(self.config['cy'])#360.5
        
    def transform(self, central_pt):
        '''
        # see https://blog.csdn.net/weixin_39568744/article/details/81034053

        Parameters
        ----------
        central_pt : list
            [u,v], cone center position in terms of camera coordinate

        Returns
        -------
        Pos_2d : list
            [X,Y], cone center position in terms of car coordinate, origin at the 
            center of two rear wheels

        '''
        # camera coordinate
        Z_cam = (self.fy * self.Y) / (central_pt[1] - self.cy)
        X_cam = (central_pt[0] - self.cx) * Z_cam / self.fx
        
        # transform camera coordinate to world coordinate
        X_car = Z_cam + self.rear_x_offset
        Y_car = - X_cam + self.rear_y_offset
        
        return [X_car, Y_car]
    
if __name__ == "__main__":
    p2c = Pixel2Car()
    print(p2c.transform([427,444]))
        
        