#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import time
import rospy
import numpy as np
import math
import rospkg
import pandas as pd
from base_link_listener import Base_Link_listener
from image_listener import ImageListener
from geometry_msgs.msg import Twist
from sklearn.neighbors import KDTree
import cv2

# this script will control the race car to navigate 
# following a path generated from simulation world
# definition, using pure-pursuit control

class Fake_navigation(object):
    def __init__(self, path_file_name = "smoothed_fake_central_path.txt",speed = 3.5, mode = "continuous", pkg_name = "auto_pilot_ml", record_control = False):
        # find to path to the current package
        self.rospack = rospkg.RosPack()
        self.pkg_name = pkg_name
        self.current_path = self.rospack.get_path(pkg_name) + "/scripts/" + path_file_name
        
        # load in path
        self.path_list = np.loadtxt(self.current_path)
        
        # create the KDTree to store the path
        self.path_tree = KDTree(self.path_list)
        
        # pure-pursuit model constants
        self.k = 0.1 # coefficient for look ahead
        self.Lfc = 4.0 # look ahead distance
        self.L = 1.6 # distance between front and rear tires
        
        self.speed = speed # constant navigation speed, default : 3.5m/s
        
        # angle for discrete control
        self.straight_angle = 0.0
        self.left_angle = 0.3
        self.right_angle = -0.3
        self.angle_tolerance = 0.125#0.1 (2 collides)
        
        # ros publisher
        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        
        # velocity object
        self.twist = Twist()
        
        # pose listener
        self.pose_listener = Base_Link_listener()
        
        # image listener, to store image
        self.img_listener = ImageListener()
        
        # control mode, continuous or discrete
        self.mode = mode
    
        # sleep time
        self.sleep_time = 0.1

        # loop closure count
        self.loop_closure_count = 0
        
        # variable for loop closure detection
        self.departured = False
        self.departure_tolerance = 2.0

        # initial pose
        self.init_pos_2d, _, _ = self.pose_listener.get_pose()

        # number of rounds
        self.number_of_rounds = 2 # go two rounds and stop, store data     
        
        # whether record image and control data
        self.record_control = record_control
        
        # variables to store data
        # data path
        self.data_path = self.rospack.get_path(pkg_name) + "/scripts/fake_navigation_data" 
        self.train_path = self.data_path + "/train_img/"
        self.valid_path = self.data_path + "/validation_img/"
        # data number count, total number of (img,control) pairs
        self.data_number_count = 0
        
        # train and validation control command
        self.train_commands = {'img_name':[], 'choice':[], 'choice_one_hot':[],'continuous':[]} # store the discrete choice, discrete choice in one hot,
        self.valid_commands = {'img_name':[], 'choice':[], 'choice_one_hot':[],'continuous':[]} # the continuous commands by pure-pursuit and img name
        self.train_count = 0
        self.valid_count = 0        
        
        
        # seperate boundary of training and validation set, using epsilon division
        self.epsilon = 0.2 # uniformly sample from [0,1], valid if sample < epsilon else train
        
        
    def get_target(self):
        # get the target for pure-pursuit control
    
        # get current position and direction in 2d
        self.pos_2d, self.direction_vector, self.yaw = self.pose_listener.get_pose()
        self.yaw = self.yaw % (2*np.pi)
        
        Lf = self.k * self.speed + self.Lfc
        
        # find the points within Lf
        ind, distance = self.path_tree.query_radius(self.pos_2d.reshape(1,-1),Lf,return_distance = True)
        
        # sort those points based on distance
        distance_order = distance[0].argsort()
        
        target = [] # empty target    
        cos_alpha = None
        
        for ii in range(len(distance_order)-1,-1,-1):
            temp_target_ind = ind[0][distance_order[ii]]
            temp_pos_2d = self.path_list[temp_target_ind]
            # check whether the points is in front of the car
            cos_alpha = np.dot(temp_pos_2d - self.pos_2d,self.direction_vector)
            if cos_alpha > 0:
                target = temp_pos_2d
                cos_alpha = cos_alpha / np.linalg.norm(temp_pos_2d - self.pos_2d)
                break
        
        return target, cos_alpha
    
    def pure_pursuit_control(self):
        
        stop = False # whether the car should stop
        choice = None # discrete control choice
        choice_one_hot = []
        delta = None
        # get the target
        target, cos_alpha = self.get_target()
        if target == []:
            print("No feasible target point found, stop vehicle!")
            self.twist.linear.x = 0.0
            self.twist.angular.z = 0.0
            stop = True
        else:
            # calculate forward distance
            Lf = self.k * self.speed + self.Lfc
            
            # calculate between current direction and target direction
            alpha = math.atan2(target[1] - self.pos_2d[1],target[0] - self.pos_2d[0]) - self.yaw
            #alpha = math.acos(cos_alpha)
            
            # calculate steer angle
            delta = math.atan2(2.0 * self.L * math.sin(alpha) / Lf, 1.0)

            print("---")            
            print("self: (%f,%f)"%(self.pos_2d[0],self.pos_2d[1]))
            print("target: (%f,%f)"%(target[0],target[1]))
            print("yaw = %f"%(self.yaw))
            print("direction:%f,%f"%(self.direction_vector[0],self.direction_vector[1]))
            print("alpha = %f"%(alpha))
            print("delta = %f"%(delta))
            
            
            self.twist.linear.x = self.speed
            
            # set output based on control mode
            if self.mode == "continuous":
                self.twist.angular.z = delta
            else:
                if abs(delta) <= self.angle_tolerance: # small angle, do not turn
                    self.twist.angular.z = 0
                    choice = 0
                    choice_one_hot = [1, 0, 0]
                elif delta > self.angle_tolerance: # large left angle
                    self.twist.angular.z = self.left_angle
                    choice = 1
                    choice_one_hot = [0, 1, 0]
                elif delta < -self.angle_tolerance: # large right angle
                    self.twist.angular.z = self.right_angle
                    choice = 2
                    choice_one_hot = [0, 0, 1]
        
        # loop closure detection
        distance_to_init = np.linalg.norm(self.pos_2d - self.init_pos_2d)
        
        if distance_to_init < self.departure_tolerance:
            # close to initial pose
            if self.departured: 
                # the vehicle had departured before, this is the next time for departure
                self.departured = False
                self.loop_closure_count += 1 # increment the number of loop closure
                
                # check whether the car should stop
                if self.loop_closure_count >= self.number_of_rounds:
                    # order the vehicle to stop
                    print("Drive through the path %d times, stop car"%(self.loop_closure_count))
                    self.twist.linear.x = 0.0
                    self.twist.angular.z = 0.0
                    stop = True
                    # check whether data needs to be stored or not
                    if self.record_control:
                        # store the control commands 
                        train_df = pd.DataFrame(self.train_commands)
                        valid_df = pd.DataFrame(self.valid_commands)
                        train_df.to_csv(self.data_path + '/train.csv')
                        valid_df.to_csv(self.data_path + '/valid.csv')
                        print("Training set size: %d, Validation set size: %d"%(self.train_count,self.valid_count))
                        
        else:
            self.departured = True
        
        # store control commands while running
        if stop != True and self.record_control: # vehicle is running, store commands required
            # add control commands to local variables
            # get image
            latest_img = self.img_listener.get_image()
            img_name = str(self.data_number_count).zfill(5) + '.jpg'
            self.data_number_count += 1
            rand_num = np.random.rand()# draw a random number between [0,1]
            if rand_num <= self.epsilon:
                # validation set
                self.valid_commands['img_name'].append(img_name)
                self.valid_commands['choice'].append(choice)
                self.valid_commands['choice_one_hot'].append(choice_one_hot)
                self.valid_commands['continuous'].append(delta)
                cv2.imwrite(self.valid_path + img_name,latest_img)
                self.valid_count += 1
            else:
                # training set
                self.train_commands['img_name'].append(img_name)
                self.train_commands['choice'].append(choice)
                self.train_commands['choice_one_hot'].append(choice_one_hot)
                self.train_commands['continuous'].append(delta)
                cv2.imwrite(self.train_path + img_name,latest_img)
                self.train_count += 1
        
        # publish control command
        self.pub.publish(self.twist)
            
        # sleep for some time, mimic the image processing and decision taking time for real controller
        time.sleep(self.sleep_time)
    
        return stop

def main():
    rospy.init_node('Fake_navigation_controller')
    fake_navigation = Fake_navigation(mode="discrete", record_control = True)
    while not rospy.is_shutdown():
        stop = fake_navigation.pure_pursuit_control()
        if stop:
            break
        #rospy.spin()

if __name__ == "__main__":
    main()
            