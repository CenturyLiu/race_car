#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 10:46:06 2020

@author: centuryliu
"""

# this script is for python 3.8

import socket
import numpy as np
import cv2
from cone_detect import DetectionClass 

class ControlServer(object):
    def __init__(self, host = '', port = 50007, original_row = 720, original_col = 1280, image_size = 640):
        self.host = host
        self.port = port
        self.buffer_size = 33554432#40960000
        self.end_note = b'finished'
        
        self.original_row = original_row
        self.original_col = original_col
        self.image_size = image_size
        
        # cone detection class
        self.cone_detection = DetectionClass()
        
        # create the server
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.bind((host, port))
        
        # wait until a client is online, start connection with the client
        self.s.listen(1)
        
        print("Control server started")
    
    def get_control_from_img(self):
        # get the image from socket client (ros node, python 2.7)
        # return control command
        
        conn, addr = self.s.accept()
        ii = 0
        with conn:
            data = b''
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
            img = np.frombuffer(data,dtype='uint8').reshape((self.original_row, self.original_col,3))
            #print(img.shape)
            cone_array = self.cone_detection.get_central_pts(img)
                
            conn.sendall(b'0') # fake control command
            conn.close()


        
        return 
    

    
if __name__ == '__main__':
    server = ControlServer()
    for ii in range(10):
        server.get_control_from_img()