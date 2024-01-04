# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 11:27:49 2023

@author: admin
"""

import cv2 
import numpy as np
import os

class FeatureExtractor:
    @classmethod
    def load_video(cls, filename, folder):
        cap = cv2.VideoCapture('./words/'+folder+'/'+filename+'.mp4')
        maxframes = 10
        frames = np.random.rand(maxframes,1,100,100)
        framecount = 0
        while(cap.isOpened()):
                # extracting the frames
                ret, img = cap.read()
                # converting to gray-scale
                if ret == True:
                    frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    if(framecount<maxframes):
                        frames[framecount]=(np.array(frame))
    
                    framecount = framecount + 1
    
                    # exiting the loop
                    key = cv2.waitKey(1)
                    if key == ord("q"):
                        break
    
            # Break the loop
                else:
                    break
        cap.release()
        return frames

    @classmethod
    def wordExtraction(cls,):
        files = os.listdir('./croppeddata')
        for filename in files:
            if(filename.split('.')[1]=='mp4'):
                cls.wordExtractionLoop(filename.split('.')[0])
        
    @classmethod
    def lipExtraction(cls,):
        files = os.listdir('./data/s1')
        for filename in files:
            if(filename.split('.')[1]=='mpg'):
                cls.lipExtractionLoop(filename.split('.')[0])
    
    @classmethod
    def wordExtractionLoop(cls, filename):
        # reading the vedio 
        words = []
        breakframe = []
        with open('./data/alignments/s1/'+filename+'.align', 'r') as f: 
            lines = f.readlines() 
            for line in lines:
                line = line.split()
                words = [*words, (line[2])]
                breakframe = [*breakframe, [int(np.floor(int(line[0])/1000)),int(np.ceil(int(line[1])/1000))]]
        
        source = cv2.VideoCapture('./croppeddata/'+filename+'.mp4') 
        if (source.isOpened()== False): 
            print("Error opening video file") 
        
        #finding resolution and fps
        fps = source.get(cv2.CAP_PROP_FPS)
        frame_width = int(source.get(3)) 
        frame_height = int(source.get(4)) 
        size = (frame_width, frame_height) 
        countn = 0
            
        for idx in range(len(words)):
            if(os.path.exists('./words/'+words[idx])==False):
                os.makedirs('./words/'+words[idx])
    
            result = cv2.VideoWriter('./words/'+words[idx]+'/'+str(idx)+filename+'.mp4',  
                        cv2.VideoWriter_fourcc(*'mp4v'), 
                        fps, size, 0) 
    
            while(source.isOpened()): 
                # cropping the frames 
                ret, img = source.read() 
                countn = countn+1
                if(np.all([ret == True, countn>=breakframe[idx][0],countn<=breakframe[idx][1]])):
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
                    result.write(gray)
                    key = cv2.waitKey(1) 
                    if key == ord("q"): 
                        break
                    
                # Break the loop 
                else: 
                    break
            # Closes all the frames 
            result.release()
        source.release() 
        # closing the window 
        cv2.destroyAllWindows() 
    
    @classmethod
    def lipExtractionLoop(cls, filename):
        mouth_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')
        
        # reading the vedio 
        source = cv2.VideoCapture('./data/s1/'+filename+'.mpg') 
        if (source.isOpened()==False):
            print("Error opening video file") 
        
        
        #finding resolution and fps
        fps = source.get(cv2.CAP_PROP_FPS)
        frame_width = int(source.get(3)) 
        frame_height = int(source.get(4)) 
        size = (frame_width, frame_height) 
        
        # video writer
        result = cv2.VideoWriter('./graydata/'+filename+'.mp4',  
                    cv2.VideoWriter_fourcc(*'mp4v'), 
                    fps, size, 0) 
          
        # Read until video is completed
        mx=0
        my=0
        mw=0
        mh=0
        n=0
        
        while(source.isOpened()): 
            # extracting the frames 
            ret, img = source.read() 
              
            # converting to gray-scale 
            if ret == True: 
                # write to gray-scale 
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        
                #cropping out mouth
                mouth = mouth_cascade.detectMultiScale(gray, 1.5, 11)
                if(len(mouth)==1):
                    mx=mx+mouth[0][0]
                    my=my+mouth[0][1]
                    mw=mw+mouth[0][2]
                    mh=mh+mouth[0][3]
                    n=n+1
                
                #writing video
                result.write(gray)
              
                # exiting the loop 
                key = cv2.waitKey(1) 
                if key == ord("q"): 
                    break
                
        # Break the loop 
            else: 
                break
        # Closes all the frames 
        source.release() 
        result.release()
    
        if(n>0):
            
            mx = int(mx/n)
            my = int(my/n)
            mw = int(mw/n)
            mh = int(mh/n)
        
            mx1 = max(mx,0)
            mx2 = min(mx + mw,frame_width)
            my1 = max(my - 10 ,0)
            my2 = min(my + mh,frame_height)
            
            # reading the gray vedio 
            sourcegray = cv2.VideoCapture('./graydata/'+filename+'.mp4') 
            if (sourcegray.isOpened()== False): 
                print("Error opening video file") 
            
            #finding resolution
            sizegray = (100,100) 
            
            # video writer
            resultgray = cv2.VideoWriter('./croppeddata/'+filename+'.mp4',  
                        cv2.VideoWriter_fourcc(*'mp4v'), 
                        25, sizegray, 0) 
            
            while(sourcegray.isOpened()): 
                # extracting the frames 
                ret, img = sourcegray.read() 
                  
                # converting to gray-scale 
                if ret == True: 
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
                    cropped = gray[my1:my2,mx1:mx2]
                    resizecropped = cv2.resize(cropped, (100, 100), interpolation = cv2.INTER_CUBIC)
                    #writing video
                    resultgray.write(resizecropped)
                  
                    # exiting the loop 
                    key = cv2.waitKey(1) 
                    if key == ord("q"): 
                        break
                    
            # Break the loop 
                else: 
                    break
            
            # Closes all the frames 
            sourcegray.release() 
            resultgray.release()
        # closing the window 
        cv2.destroyAllWindows() 
    
