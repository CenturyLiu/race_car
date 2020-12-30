#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 14:49:41 2020

@author: centuryliu
"""

from cone_detect import DetectionClass
from pixel2car import Pixel2Car
from pure_pursuit_wrapper import PurePursuit
#from generate_path import sort_path_pt
from generate_path2 import link_path_pt
import socket
import numpy as np
from sklearn.neighbors import KDTree
import cv2


class NaiveControlServer(object):
    def __init__(self, host = '', port = 50007, original_row = 720, original_col = 1280, image_size = 640, distance = 16.0, speed = 3.5, Lfc = 4.0, debug = False):
        self.host = host
        self.port = port
        self.buffer_size = 33554432#40960000
        self.end_note = b'finished'
        
        self.original_row = original_row
        self.original_col = original_col
        self.image_size = image_size
        
        self.speed = speed
        self.Lfc = Lfc
        
        # cone detection class
        self.cone_detection = DetectionClass()
        
        # pixel coordinate transform class
        self.pixel_2_car = Pixel2Car()
        
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
            
        
        # create the server
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.bind((host, port))
        
        
        # wait until a client is online, start connection with the client
        self.s.listen(1)
        
        print("Control server started")
        
    
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
                
        tree = KDTree(choice_tree)
    
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
        # get the image from socket client (ros node, python 2.7)
        # send control command to socket client
        # return whether this node should be shut down
        self.iteration_count += 1
        # get image
        conn, addr = self.s.accept()
        data = b''
        stop = False
        with conn:
            while True:
                try: 
                    chunk = conn.recv(self.buffer_size)
                    data += chunk
                    if data[-8:] == self.end_note:
                        break
                except socket.error:
                    conn.close()
                    return
                    
            
            data = data[:-8]
            
            if data == b'':
                print("No image received, stop car")
                conn.sendall(np.array([0.0,0.0]).tobytes()) # speed, control
                return True # shut node down
                
            
            img = np.frombuffer(data,dtype='uint8').reshape((self.original_row, self.original_col,3))
            #print(img.shape)
            
            # get cone center pos in terms of image coordinate
            red_pts, blue_pts = self.cone_detection.get_central_pts(img)
            
            # change coordinate into car frame
            red_pts_car = []
            blue_pts_car = []
            
            red_x = []
            red_y = []
            blue_x = []
            blue_y = []
            
            for red_pt in red_pts:
                central_pt = self.pixel_2_car.transform(red_pt)
                if central_pt[0]**2 + central_pt[1]**2 < self.distance_2: # cone close enough to be considered
                    red_pts_car.append(np.array(central_pt))
                    red_x.append(central_pt[0])
                    red_y.append(central_pt[1])
            for blue_pt in blue_pts:
                central_pt = self.pixel_2_car.transform(blue_pt)
                if central_pt[0]**2 + central_pt[1]**2 < self.distance_2: # cone close enough to be considered
                    blue_pts_car.append(np.array(central_pt))
                    blue_x.append(central_pt[0])
                    blue_y.append(central_pt[1])
            
            if self.debug:
                print("---")
                print("iteration %d", self.iteration_count)
                '''
                if len(red_x) != 0:
                    mr, br = np.polyfit(red_x, red_y, 1)
                    print("red:")
                                        for pt in red_pts_car:
                        img = cv2.circle(img, (int(img_size / 2) - int(pt[1]), int(img_size) - int(pt[0])), 2, (0,0,255))
                    
                    for pt in blue_pts_car:
                        img = cv2.circle(img, (int(img_size / 2) - int(pt[1]), int(img_size) - int(pt[0])), 2, (255,0,0))
                    
                    cv2.imshow('path',img)
                    cv2.waitKey(1)print("slope = %f, intercept = %f"%(mr,br))
                if len(blue_x) != 0:
                    mb, bb = np.polyfit(blue_x, blue_y, 1)
                    print("blue:")
                    print("slope = %f, intercept = %f"%(mb,bb))
                '''
                print("red: %d"%(len(red_pts_car)))
                print("blue: %d"%(len(blue_pts_car)))
            
            '''
            # if too few cones are detected, enlarge the distance and loop again
            if len(red_pts) <= 2:
                distance_2 = (self.distance + 5) ** 2
                red_pts_car = []
                for red_pt in red_pts:
                    central_pt = self.pixel_2_car.transform(red_pt)
                    if central_pt[0]**2 + central_pt[1]**2 < distance_2: # cone close enough to be considered
                        red_pts_car.append(np.array(central_pt))
            
            if len(blue_pts) <= 2:
                distance_2 = (self.distance + 5) ** 2
                blue_pts_car = []
                for blue_pt in blue_pts:
                    central_pt = self.pixel_2_car.transform(blue_pt)
                    if central_pt[0]**2 + central_pt[1]**2 < distance_2: # cone close enough to be considered
                        blue_pts_car.append(np.array(central_pt))
            '''
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
                    path_tree = KDTree(total_path)
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
                
            conn.sendall(np.array([speed, steer]).tobytes()) # fake control command
            conn.close()
            
        return stop

if __name__ == "__main__":
    # for training_track.launch
    #control_server = NaiveControlServer(speed = 3.5, Lfc = 4.0, distance = 16.0, debug = True)
    control_server = NaiveControlServer(speed = 4.0, Lfc = 4.0, distance = 16.0, debug = True)
    while True:
        stop = control_server.get_control_from_img()
        if stop:
            print("shutdown node")
            break