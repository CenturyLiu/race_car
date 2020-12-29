#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Created on Mon Dec 28 16:10:59 2020

@author: centuryliu
"""


import socket
import numpy as np

from image_listener import ImageListener
from geometry_msgs.msg import Twist
import rospy

class NaiveNavigation(object):
    def __init__(self, host = '', port = 50007):
        # ros publisher
        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        
        # velocity object
        self.twist = Twist()
        
        # image listener, to store image
        self.img_listener = ImageListener()
        
        self.host = host
        self.port = port
        self.buffer_size = 33554432#40960000
        self.end_note = b'finished'
        print("Naive navigation client initialized")
        
        self.count = 0
        self.max_step = 10000
        
    def get_control_from_img(self):
        # get image
        latest_img = self.img_listener.get_image()
        
        # send image to navigation server, get control command
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((self.host, self.port))
        img = np.asarray(latest_img)
        
        s.sendall(img.tobytes() + self.end_note)
        command = s.recv(32) # raw control command  
        s.close()
        
        
        # recover the command
        command = np.frombuffer(command)
        print("Command: speed = %f, steer = %f" %(command[0],command[1]))
        self.twist.linear.x = command[0]
        self.twist.angular.z = command[1]
        
        # send the command
        self.pub.publish(self.twist)
        
        self.count += 1
        if self.count >= self.max_step:
            return True
        return False

def main():
    rospy.init_node('Naive_navigation_controller')
    #fake_navigation = Fake_navigation(mode="discrete", record_control = True)
    naive_navigation = NaiveNavigation()
    while not rospy.is_shutdown():
        stop = naive_navigation.get_control_from_img()
        if stop:
            print("stop navigation")
            break
        #rospy.spin()

if __name__ == "__main__":
    main()
