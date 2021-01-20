#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 15:50:07 2021

@author: centuryliu
"""
from configobj import ConfigObj

# write all parameters
def writeParam(filename):
    config = ConfigObj()
    config.filename = filename
    
    config["img_x"] = 1280
    config["img_y"] = 640
    
    config.write()
    

def readParam(filename):
    config = ConfigObj(filename)
    
    print(config['image_topic'])
    print(config['vehicle_control_topic'])
    

if __name__ == "__main__":
    #writeParam("test.param")
    readParam("../params/racecar_params")