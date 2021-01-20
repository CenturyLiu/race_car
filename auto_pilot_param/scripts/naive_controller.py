#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 13:28:46 2021

@author: centuryliu
"""

import socket
import numpy as np
from sklearn.neighbors import KDTree
from generate_path2 import link_path_pt
from image_listener import ImageListener
from geometry_msgs.msg import Twist
import rospy
from pure_pursuit_wrapper import PurePursuit
import cv2
import rospkg
from configobj import ConfigObj

class NaiveController(object):
    def __init__(self, host = '', port = 50008, distance = 16.0, speed = 3.5, Lfc = 4.0, debug = False):
        # get directory to param file
        self.rospack = rospkg.RosPack()
        self.param_file = self.rospack.get_path('auto_pilot_param') + '/params/racecar_params' 
        
        # get params
        self.config = ConfigObj(self.param_file)
        
        # ros publisher
        self.pub = rospy.Publisher(self.config['vehicle_control_topic'], Twist, queue_size=1)
        
        # velocity object
        self.twist = Twist()
        
        # image listener, to store image
        self.img_listener = ImageListener()
        
        # control speed and look ahead distance
        self.speed = speed
        self.Lfc = Lfc
        
        # purepursuit wrapper class
        self.purepursuit = PurePursuit(speed = self.speed, Lfc = self.Lfc)
        
        # consider cones within distance
        self.distance = distance
        self.distance_2 = distance**2
        self.neighbor_distance = 1.0
        
        # max tolerance between 2 continuous angles
        self.max_steer_change = 0.45#0.3
        
        self.left_angle = 0.35
        self.right_angle = -0.35#-0.35
        
        self.left_angle_large = 0.5#0.6
        self.right_angle_large = -0.5#-0.6
        
        self.left_large_count = 0
        self.right_large_count = 0
        
        self.no_feasible_count = 0
        self.no_feasible_limit = 5 # allow in maximum 5 continuous times the controller cannot find target
        self.misclassify_limit = 2
        #self.last_not_feasible = False 
        self.last_steer = 0.0
        self.last_speed = self.speed
        
        
        
        self.no_cone_count = 0
        
        self.iteration_count = 0
        
        # draw the path
        self.debug = debug
        
        
        self.host = host
        self.port = port
        self.buffer_size = 33554432#40960000
        self.end_note = b'finished'
        print("Naive slam client initialized")
        
        self.count = 0
        self.max_step = 10000
        
    def get_cone_pos_from_img(self):
        # get image
        latest_img = self.img_listener.get_image()
        
        # send image to navigation server, get control command
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((self.host, self.port))
        img = np.asarray(latest_img)
        
        s.sendall(img.tobytes() + self.end_note)
        
        # receive cone pos
        data = b''
        while True:
            try: 
                chunk = s.recv(self.buffer_size)
                data += chunk
                if data[-8:] == self.end_note:
                    s.close()
                    break
            except socket.error:
                s.close()
                return
        
        
        #command = s.recv(32) # raw control command  
        #s.close()
        
        
        # recover the command
        command = data[:-8]
        cone_pts = np.frombuffer(command).reshape(-1,3)
        #print("Command %d: speed = %f, steer = %f" %(self.count ,command[0],command[1]))
        red_pts_car = []
        blue_pts_car = []
        
        
        
        for pts in cone_pts:
            if pts[2] == 0.0:
                red_pts_car.append(np.array([pts[0], pts[1]]))
            else:
                blue_pts_car.append(np.array([pts[0], pts[1]]))
        
        
        return red_pts_car, blue_pts_car
        
    def generate_central_path(self, red_pts_car, blue_pts_car):
        len_red = len(red_pts_car)
        len_blue = len(blue_pts_car)
        # generate path by using avg
        if len_red >= len_blue:
            choice_tree = red_pts_car
            choice_not_tree = blue_pts_car
        else:
            choice_tree = blue_pts_car
            choice_not_tree = red_pts_car
                
        tree = KDTree(np.array(choice_tree))
    
        central_path_pts = []
        for pt in choice_not_tree:
            _, ind = tree.query(pt.reshape(1,-1),2) # get the nearest 2 other color cone
            neighbor1 = choice_tree[ind[0][0]]
            central_path_pts.append((neighbor1 + pt) / 2)
            neighbor2 = choice_tree[ind[0][1]]
            #if np.linalg.norm(neighbor1 - neighbor2) < self.neighbor_distance: # nearest neighbors too close
            #    if len(choice_tree) >= 3:
            #        _, ind = tree.query(pt.reshape(1,-1),3)
            #        neighbor2 = choice_tree[ind[0][2]]
            central_path_pts.append((neighbor2 + pt) / 2)
        # sort the central path pts
        central_pt_2d = np.array(central_path_pts)
        total_path = link_path_pt(central_pt_2d)
                
        # add [0,0] to total_path
        if np.linalg.norm(np.array([0,0]) - total_path[0]) > np.linalg.norm(np.array([0,0]) - total_path[-1]):
            # reverse the order
            total_path.reverse()
        return total_path

    def get_control_from_img(self):
        red_pts_car_raw, blue_pts_car_raw = self.get_cone_pos_from_img()
        
        
        red_pts_car = []
        blue_pts_car = []
            
        red_x = []
        red_y = []
        blue_x = []
        blue_y = []
        
        for central_pt in red_pts_car_raw:
            if central_pt[0]**2 + central_pt[1]**2 < self.distance_2: # cone close enough to be considered
                red_pts_car.append(np.array(central_pt))
                red_x.append(central_pt[0])
                red_y.append(central_pt[1])
        for central_pt in blue_pts_car_raw:
            if central_pt[0]**2 + central_pt[1]**2 < self.distance_2: # cone close enough to be considered
                blue_pts_car.append(np.array(central_pt))
                blue_x.append(central_pt[0])
                blue_y.append(central_pt[1])
        
        
        if self.debug:
            print("---")
            print("iteration %d" %(self.count))
            print("red: %d"%(len(red_pts_car)))
            print("blue: %d"%(len(blue_pts_car)))
        
        
        # generate path and get control commands
            
        steer = 0.0
        speed = self.speed
            
        len_red = len(red_pts_car)
        len_blue = len(blue_pts_car)
            
        # try to form central path
        total_path = []
        if len_red >= 2 and len_blue >= 2:

            total_path = self.generate_central_path(red_pts_car, blue_pts_car)
                
            total_path.insert(0,np.array([0,0]))
            if len(total_path) <= 3:
                temp = []
                for ii in range(1,len(total_path)):
                    temp.append(total_path[ii-1])
                    temp.append((total_path[ii-1] + total_path[ii]) / 2)
                temp.append(total_path[-1])
                total_path = temp
                
            else:
                # check whether the central path is "classifying" the red and blue cones
                path_tree = KDTree(np.array(total_path))
                misclassify = 0
                for pt in red_pts_car:
                    _, ind = path_tree.query(pt.reshape(1,-1),1)
                    index = ind[0][0]
                    if index != len(total_path) - 1:
                        vec_path = np.array([total_path[index + 1][0] - total_path[index][0],total_path[index + 1][1] - total_path[index][1], 0])
                    else:
                        vec_path = np.array([total_path[index][0] - total_path[index - 1][0],total_path[index][1] - total_path[index - 1][1], 0])
                    
                    vec_2 = np.array([total_path[index][0] - pt[0], total_path[index][1] - pt[1],0])
                    cross = np.cross(vec_2, vec_path)
                    if cross[2] < 0:
                        misclassify += 1
                
                for pt in blue_pts_car:
                    _, ind = path_tree.query(pt.reshape(1,-1),1)
                    index = ind[0][0]
                    if index != len(total_path) - 1:
                        vec_path = np.array([total_path[index + 1][0] - total_path[index][0],total_path[index + 1][1] - total_path[index][1], 0])
                    else:
                        vec_path = np.array([total_path[index][0] - total_path[index - 1][0],total_path[index][1] - total_path[index - 1][1], 0])
                        
                    vec_2 = np.array([total_path[index][0] - pt[0], total_path[index][1] - pt[1],0])
                    cross = np.cross(vec_2, vec_path)
                    if cross[2] > 0:
                        misclassify += 1
                
                if self.debug:
                    print("misclassification : %d / %d"%(misclassify, len(red_pts_car) + len(blue_pts_car)))
                    
                if misclassify >= self.misclassify_limit :
                    # too much misclassification, use smaller radius, re-construct central path
                    red_pts_car_temp = []
                    blue_pts_car_temp = []
                    distance_2 = (self.distance - 5) ** 2
                    for pt in red_pts_car:
                        if pt[0] ** 2 + pt[1] ** 2 <= distance_2:
                            red_pts_car_temp.append(pt)
                    
                    for pt in blue_pts_car:
                        if pt[0] ** 2 + pt[1] ** 2 <= distance_2:
                            blue_pts_car_temp.append(pt)
                    
                    red_pts_car = red_pts_car_temp
                    blue_pts_car = blue_pts_car_temp
                    len_red = len(red_pts_car)
                    len_blue = len(blue_pts_car)
                    if len_red >= 2 and len_blue >= 2:
                        total_path = self.generate_central_path(red_pts_car, blue_pts_car)
                        total_path.insert(0,np.array([0,0]))
                        if len(total_path) <= 3:
                            temp = []
                            for ii in range(1,len(total_path)):
                                temp.append(total_path[ii-1])
                                temp.append((total_path[ii-1] + total_path[ii]) / 2)
                            temp.append(total_path[-1])
                            total_path = temp
                        
            
        # draw cone in range and central path
        if self.debug:
            img_size = 32
            img = np.ones((img_size,img_size,3), np.uint8) * 255
            
            
        # start control
        if len_red >= 2 and len_blue >= 2:
                
            # draw the path
                
            if self.debug:
                    
                for ii in range(0,len(total_path) - 1):
                    img = cv2.line(img,(int(img_size / 2) - int(total_path[ii][1]),int(img_size) - int(total_path[ii][0])),(int(img_size / 2) - int(total_path[ii+1][1]),int(img_size) - int(total_path[ii+1][0])),(0,255,0), thickness = 2)
                    
                
            ret_val = self.purepursuit.get_speed_steer(total_path, current_xy = [0.0,0.0], current_yaw = 0.0)
            speed = ret_val[0]
            steer = ret_val[1]
            if speed == 0.0: # pure pursuit wrapper failed to find target
                print(total_path)
                self.no_feasible_count += 1
                if self.no_feasible_count > self.no_feasible_limit:
                    speed = 0.0
                    steer = 0.0
                else:
                    speed = self.last_speed
                    steer = self.last_steer
            else:
                self.no_feasible_count = 0
                if abs(steer - self.last_steer) > self.max_steer_change:
                    # steer change too much, abandon data
                    steer = self.last_steer
                    
                if steer > 0.18 and steer < 0.3:
                    steer = 0.3
                elif steer < -0.18 and steer > -0.3:
                    steer = -0.3
                    
                    
                    
                self.last_speed = speed
                self.last_steer = steer
                self.no_cone_count = 0
                
        elif len_red >= 3 and len_blue == 1:
            print("Turn right, slow down")
            # turn right
            steer = self.right_angle
            speed = self.speed / 2 # slow down
            self.no_cone_count = 0
            self.last_speed = speed
            self.last_steer = steer
                
        elif len_blue >= 3 and len_red == 1:
            print("Turn left, slow down")
            # turn left
            steer = self.left_angle
            speed = self.speed / 2 # slow down
            self.no_cone_count = 0
        elif len_red != 0 and len_blue == 0:
            print("No blue, turn right immediately")
            steer = self.right_angle_large
            speed = self.speed / 3
            self.no_cone_count = 0
            self.last_speed = speed
            self.last_steer = steer
        elif len_red == 0 and len_blue != 0:
            print("No red, turn left immediately")
            # turn left
            steer = self.left_angle_large
            speed = self.speed / 3 # slow down
            self.no_cone_count = 0
            self.last_speed = speed
            self.last_steer = steer
        else:
            if self.no_cone_count >= self.no_feasible_limit:
                # too few cones in range, stop
                print("Too few input pts! Stop!")
                steer = 0.0
                speed = 0.0
                stop = True
            else:
                # allow the car follow original path for some time
                steer = self.last_steer
                speed = self.last_speed
                self.no_cone_count += 1
                print("Too few input pts!")
            
            
        if self.debug:
            for pt in red_pts_car:
                img = cv2.circle(img, (int(img_size / 2) - int(pt[1]), int(img_size) - int(pt[0])), 2, (0,0,255))
                    
            for pt in blue_pts_car:
                img = cv2.circle(img, (int(img_size / 2) - int(pt[1]), int(img_size) - int(pt[0])), 2, (255,0,0))
                    
            cv2.imshow('path',img)
            cv2.waitKey(1)
            print("speed: %f, steer: %f"%(speed,steer))
        
        self.twist.linear.x = speed
        self.twist.angular.z = steer
        
        # send the command
        self.pub.publish(self.twist)
        
        self.count += 1
        if self.count >= self.max_step:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((self.host, self.port))
            s.sendall(self.end_note)
            command = s.recv(32)
            s.close()
            self.twist.linear.x = 0.0
            self.twist.angular.z = 0.0
            self.pub.publish(self.twist)
            
            return True
        return False
        

def main():
    rospy.init_node('Naive_control_client')
    #fake_navigation = Fake_navigation(mode="discrete", record_control = True)
    naive_navigation = NaiveController(speed = 5.0, Lfc = 4.0, distance = 16.0, debug = True)
    while not rospy.is_shutdown():
        stop = naive_navigation.get_control_from_img()
        if stop:
            print("stop navigation")
            break
        #rospy.spin()

if __name__ == "__main__":
    main()
