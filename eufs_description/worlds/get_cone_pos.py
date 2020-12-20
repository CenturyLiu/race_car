#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import copy
import matplotlib.pyplot as plt
from tf.transformations import quaternion_matrix

def quaternion_to_matrix(a,b,c,d):
    '''
    R = np.array([[a**2 + b**2 - c**2 - d**2, 2*b*c - 2*a*d, 2*b*d + 2*a*c],
                  [2*b*c + 2*a*d, a**2 - b**2 + c**2 - d**2, 2*c*d - 2*a*b],
                  [2*b*d - 2*a*c, 2*c*d + 2*a*b, a**2 - b**2 - c**2 + d**2],])
    '''
    R = quaternion_matrix([a,b,c,d])[0:3,0:3]#quaternion_matrix([a,b,c,d])[0:3,0:3]
    #print(quaternion_matrix([a,b,c,d]))
    return R
    

class Get_cone_pos(object):
    def __init__(self, state_file_name = "model_state.txt"):
        self.state_file_name = state_file_name
        self.red_cone_list = []
        self.blue_cone_list = []
        self.model_names = []
        self.model_pose = []
        
        # local transformation model
        self.straight_track_blue = np.array([[5.00125, 1.5533, 0],
                                             [0.001248, 1.5533, 0],
                                             [-4.99875, 1.5533, 0]]).T
        self.straight_track_red = np.array([[5.00125, -1.57663, 0],
                                            [-0.067192, -1.52376, 0],
                                            [-4.9378, -1.55952, 0]]).T
        
        self.left_track_blue = np.array([[4.01224, 3.56439, 0],
                                         [1.07158, 2.90676, 0],
                                         [-1.63633, 1.45358, 0],
                                         [-2.87212, -0.307147, 0],
                                         [-3.90319, -4.54641, 0],
                                         [-3.58525, -2.23325, 0],]).T
                                    
        self.left_track_red = np.array([[4.09681, 0.453585, 0],
                                        [1.9348, -0.231588, 0],
                                        [-0.900464, -4.54641, 0],
                                        [0.33919, -1.92026, 0],]).T
        
        self.right_track_blue = np.array([[7.02182, -0.020955, 0],
                                          [2.01267, -5.02095, 0],
                                          [3.30371, -2.38642, 0],
                                          [4.83695, -0.737396, 0]]).T
                                    
        self.right_track_red = np.array([[6.86979, 3.10588, 0],
                                         [4.45799, 2.73997, 0],
                                         [2.70546, 1.86013, 0],
                                         [1.29777, 0.979045, 0],
                                         [0.020348, -0.677548, 0],
                                         [-0.703674, -2.69605, 0],
                                         [-0.979653, -5.02095, 0],]).T        
        
        
    def read_file(self):
        self.f = open(self.state_file_name, "r")
        
        line_list = []
        for line in self.f:
            line_list.append(line)

        segment_line = 0        
        
        # get all the names
        for ii in range(len(line_list)):
            line = line_list[ii]
            self.model_names += line.split(",")
            if "]" in line:
                segment_line = ii
                break
            #print(line)
        #print(self.model_names)
        #print(segment_line)
                
        # get all pos
        temp_set = []
        for ii in range(segment_line + 1,len(line_list)):
            line = line_list[ii]
            if "pose" in line:
                continue
            
            elif "- " in line:
                if temp_set != []:
                    self.model_pose.append(temp_set)
                temp_set = []
                continue
            
            elif "twist" in line:
                self.model_pose.append(temp_set)
                temp_set = []
                break
            
            else:
                temp_set.append(line)
            
        #print(self.model_pose)
    def get_cones(self):
        count = 0
        for model_name in self.model_names:
            if "straight_track" in model_name:
                red = copy.copy(self.straight_track_red)
                blue = copy.copy(self.straight_track_blue)
            elif "left_turn" in model_name:
                red = copy.copy(self.left_track_red)
                blue = copy.copy(self.left_track_blue)
            elif "right_turn" in model_name:
                red = copy.copy(self.right_track_red)
                blue = copy.copy(self.right_track_blue)
            else:
                if "[" in model_name or "]" in model_name:
                    count += 1
                continue
            
            trans, R = self.get_individual_pos(self.model_pose[count])
            #print(self.straight_track_red)
            #print(self.left_track_red)
            #print(self.right_track_red)
            
            red = trans + np.dot(R, red)
            blue = trans + np.dot(R,blue)
            
            red = red.T
            blue = blue.T
            for pt in red:
                self.red_cone_list.append(pt)
            
            for pt in blue:
                self.blue_cone_list.append(pt)
                        
            count += 1
        return self.red_cone_list, self.blue_cone_list

    
    def get_individual_pos(self,individual_pose):
        x_line = individual_pose[1].split(": ")[1]
        x = float(x_line.split("\n")[0])
        
        y_line = individual_pose[2].split(": ")[1]
        y = float(y_line.split("\n")[0])
        
        qx_line = individual_pose[5].split(": ")[1]
        qx = float(qx_line.split("\n")[0])
        
        qy_line = individual_pose[6].split(": ")[1]
        qy = float(qy_line.split("\n")[0])
        
        qz_line = individual_pose[7].split(": ")[1]
        qz = float(qz_line.split("\n")[0])
        
        qw_line = individual_pose[8].split(": ")[1]
        qw = float(qw_line.split("\n")[0])
        
        return np.array([[x],[y],[0]]), quaternion_to_matrix(qx,qy,qz,qw)
        
    def visualize(self):
        red = np.array(self.red_cone_list).T
        blue = np.array(self.blue_cone_list).T
        plt.plot(red[0],red[1],'ro',blue[0],blue[1],'bo')
        plt.show()
            
if __name__ == "__main__":
    get_clone_pos = Get_cone_pos()
    get_clone_pos.read_file()
    get_clone_pos.get_cones()
    get_clone_pos.visualize()