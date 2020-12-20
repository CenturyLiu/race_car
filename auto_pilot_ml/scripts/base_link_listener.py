#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import, division, print_function
import os
import sys

import numpy as np
from tf.transformations import euler_from_quaternion
from gazebo_msgs.msg import LinkStates
import time
import rospy

class Base_Link_listener(object):
    def __init__(self):
        self.latest_msg = rospy.wait_for_message('/gazebo/link_states',LinkStates)
        self.link_sub = rospy.Subscriber('/gazebo/link_states',LinkStates,self.link_callback)
    
    def link_callback(self,msg):
        self.latest_msg = msg
        
    def get_pose(self):
        name_list = self.latest_msg.name
        index = None
        if "eufs::base_footprint" in name_list:
            index = name_list.index("eufs::base_footprint")
            #print("%d"%(index))
        
            pose = self.latest_msg.pose[index]
            x = pose.position.x
            y = pose.position.y
            
            quaternion = (pose.orientation.x,
                          pose.orientation.y,
                          pose.orientation.z,
                          pose.orientation.w)
            
            euler = euler_from_quaternion(quaternion)
            
            direction_vec = np.array([np.cos(euler[2]),np.sin(euler[2])]) # unit direction vec, [cos(yaw),sin(yaw)]
            return np.array([x,y]), direction_vec, euler[2]
        
        return [], []
            
        

if __name__ == "__main__":
    rospy.init_node('base_link_listener')
    base_link_listener = Base_Link_listener()
    while not rospy.is_shutdown():
        pose_2d, direction_vec, yaw =  base_link_listener.get_pose()
        print(pose_2d)
        time.sleep(0.1)