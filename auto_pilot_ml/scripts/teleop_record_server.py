#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import, division, print_function

from auto_pilot_ml.srv import teleop_record
from geometry_msgs.msg import Twist
import rospy
import math
import time

class Teleop_record_server(object):
    def __init__(self):
        # velocity publisher
        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        
        # steer angles
        self.steer_list = [0.0, -math.atan2(16,65), math.atan2(16,65)] # forward, left, right
        
        # velocity object
        self.twist = Twist()
        
        # sleep time
        self.sleep_time = 0.1
        
    def teleop_record_func(self, req):
        # function for handling request and give 
        ret_str = ""    
    
        choice = req.choice
        if choice != 0 and choice != 1 and choice != 2:
            print("Stop car!")
            self.twist.linear.x = 0.0
            self.twist.angular.z = 0.0
            ret_str = "Vehicle stopped!"
        else:
            self.twist.angular.z = self.steer_list[int(choice)]
            self.twist.linear.x = req.line_vel
            if choice == 0:
                ret_str = "Go straight"
            elif choice == 1:
                ret_str = "Turn left"
            else:
                ret_str = "Turn right"
        
        # publish the velocity
        self.pub.publish(self.twist)
        
        # sleep for specified time
        time.sleep(self.sleep_time)
        
        # return the ret_str
        return ret_str
            
        

def teleop_record_server():
    # start the server for teleop-record     
    rospy.init_node('teleop_record_server')
    record_server = Teleop_record_server()
    s = rospy.Service('teleop_record_service', teleop_record, record_server.teleop_record_func)
    print("Ready to teleop the vehicle with straight, left and right")
    rospy.spin()

if __name__ == "__main__":
    teleop_record_server()