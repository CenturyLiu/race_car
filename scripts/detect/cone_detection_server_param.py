#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 13:33:11 2021

@author: centuryliu
"""


from cone_detect import DetectionClass
from pixel2car_param import Pixel2Car

import socket
import numpy as np

from configobj import ConfigObj


class ConeDetectionServer(object):
    def __init__(self, host = '', port = 50008):
        
        # get parameters
        
        self.config = ConfigObj('../../auto_pilot_param/params/racecar_params')
        
        
        self.host = host
        self.port = port
        self.buffer_size = 33554432#40960000
        self.end_note = b'finished'
        
        self.original_row = int(self.config['original_row'])#original_row
        self.original_col = int(self.config['original_col'])#original_col
        self.image_size = int(self.config['image_size'])#image_size
        
        
        # cone detection class
        self.cone_detection = DetectionClass(weight_file=self.config['weight_file'])
        
        # pixel coordinate transform class
        self.pixel_2_car = Pixel2Car()
            
        
        # create the server
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.bind((host, port))
        
        
        # wait until a client is online, start connection with the client
        self.s.listen(1)
        
        print("Control server started")
        
    
        
    def get_control_from_img(self):
        # get the image from socket client (ros node, python 2.7)
        # send cone pose to socket client
        # return whether this node should be shut down
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
                conn.sendall(self.end_note) 
                return True # shut node down
                
            
            img = np.frombuffer(data,dtype='uint8').reshape((self.original_row, self.original_col,3))
            #print(img.shape)
            
            # get cone center pos in terms of image coordinate
            red_pts, blue_pts = self.cone_detection.get_central_pts(img)
            
            # change coordinate into car frame
            pts_car = []

            
            for red_pt in red_pts:
                central_pt = self.pixel_2_car.transform(red_pt)
                central_pt.append(0.0) # 0.0 represents red
                pts_car.append(np.array(central_pt))

            for blue_pt in blue_pts:
                central_pt = self.pixel_2_car.transform(blue_pt)
                central_pt.append(1.0) # 1.0 represents blue
                pts_car.append(np.array(central_pt))

            
            cone_data = np.array(pts_car).tobytes() + self.end_note
            conn.sendall(cone_data) # send detected cones
            conn.close()
            
        return stop

if __name__ == "__main__":
    # for training_track.launch
    #control_server = NaiveControlServer(speed = 3.5, Lfc = 4.0, distance = 16.0, debug = True)
    control_server = ConeDetectionServer()
    while True:
        stop = control_server.get_control_from_img()
        if stop:
            print("shutdown node")
            break