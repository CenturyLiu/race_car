#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 13:57:24 2021

@author: centuryliu
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline

class Display(object):
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.rotation_matrix = np.array([[0,-1],[1,0]])
        
    def path_display(self, red_map_pts, blue_map_pts, path_pts):
        # clear the axis
        self.ax.clear()
        
        
        # plot red/blue cones
        if len(red_map_pts) != 0:
            red_pts = red_map_pts.T
        
            red_pts = np.dot(self.rotation_matrix,red_pts)
        
            self.ax.plot(red_pts[0],red_pts[1],'or')
            
        if len(blue_map_pts) != 0:    
            blue_pts = blue_map_pts.T
            blue_pts = np.dot(self.rotation_matrix,blue_pts)
            self.ax.plot(blue_pts[0],blue_pts[1],'ob')
        
        if len(path_pts) != 0:
            # plot planned path
            #path_pts = path_pts.T
            #path_pts = np.dot(self.rotation_matrix,path_pts)
            
            
            # path smoothing
            distance = np.cumsum( np.sqrt(np.sum( np.diff(path_pts,axis=0)**2, axis = 1)))
            distance = np.insert(distance, 0, 0)/distance[-1]
        
            # Build a list of the spline function, one for each dimension:
            splines = [UnivariateSpline(distance, coords, k=3, s=.2) for coords in np.array(path_pts).T]
        
            alpha = np.linspace(0,1.0, 20 * len(distance))
            smoothed_path = np.vstack( [spl(alpha) for spl in splines] ).T
            
            smoothed_path = np.dot(self.rotation_matrix, smoothed_path.T)
            
            self.ax.plot(smoothed_path[0],smoothed_path[1],'g')
        
        # plot current pose, which is fixed
        self.ax.quiver(0,0,0,1, color = 'y', hatch = '.')
        
        # display
        plt.show(block=False)
        plt.pause(0.001)
    
    def slam_display(self, new_pose, red_map_pts, blue_map_pts, particles):
        '''
        

        Parameters
        ----------
        new_pose : np.array([x,y,yaw])
            the pose of a point.
        red_map_pts : np.array([[x,y],[x,y],...])
            red cone pose in n*2 format, map coordinate
        blue_map_pts : np.array([[x,y],[x,y],...])
            blue pose in n*2 format, map coordinate
        particles : np.array([[x1,x2,...],[y1,y2,...],[yaw1,yaw2,...]])
            particles, 3*n

        Returns
        -------
        None.

        '''
        # clear the axis
        self.ax.clear()
        
        
        # plot red/blue cones
        red_pts = red_map_pts.T
        blue_pts = blue_map_pts.T
        self.ax.plot(red_pts[0],red_pts[1],'or')
        self.ax.plot(blue_pts[0],blue_pts[1],'ob')
        
        # plot new pose
        x_pose = np.array([new_pose[0]])
        y_pose = np.array([new_pose[1]])
        u_pose = np.array([np.cos(new_pose[2])])
        v_pose = np.array([np.sin(new_pose[2])])
        
        self.ax.quiver(x_pose,y_pose,u_pose,v_pose, color = 'y', hatch = '.')
        
        # plot particles
        self.ax.quiver(particles[0],particles[1],np.cos(particles[2]),np.sin(particles[2]), color='g',hatch = '.')

        
        # display
        plt.show(block=False)
        plt.pause(0.001)
        