import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json

from infer_segmentation import Infer

with open("config.json") as json_config_file:
    config = json.load(json_config_file)

VideoPath = config["video_path"]

class Segementation():

    def __init__(self, classes_dict, classes_to_train, model, backbone, path_to_model, image_shape=[716,1024]):

        self.gtf = Infer()
        self.gtf.Data_Params(classes_dict, classes_to_train, image_shape=image_shape)
        self.gtf.Model_Params(model=model, backbone=backbone, path_to_model=path_to_model)
        self.gtf.Setup()

    def create_video(self, output_video_path, size, frame_rate, img_array):
        out = cv2.VideoWriter(output_video_path,cv2.VideoWriter_fourcc(*'DIVX'), frame_rate, size)
 
        for i in range(len(img_array)):
            out.write(img_array[i])
        
    def segment_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        img_array = []
        count = 1
        while(cap.isOpened()):                    # play the video by reading frame by frame
            ret, frame = cap.read()
            if ret==False:
                print("Video has ended")
                break
                # get dimensions of image
            dimensions = frame.shape
            
            # height, width, number of channels in image
            height = frame.shape[0]
            width = frame.shape[1]
            channels = frame.shape[2]
            

            cv2.imwrite('frame.png', frame)

            pr_mask, img_list = self.gtf.Predict('frame.png', vis = False)

            hand = [list(map(int,i)) for i in  img_list[1]]
            hand = [[j*255 for j in i] for i in hand]
            hand = np.array(hand)

        
            img_hand = hand.astype(np.uint8)
            
            boxed_frame = self.add_box(frame, img_hand, count)

            img_array.append(boxed_frame)

            count += 1

            if cv2.waitKey(1) == ord('q'):
                break

        cap.release()
        img = cv2.imread("frame.png")
        height, width, layers = img.shape
        size = (width,height)
        self.create_video('output.avi', size , 10, img_array)

    def add_box(self, original_image, segmented_image, num):


        height = original_image.shape[0]
        width = original_image.shape[1]
        segmented_image = cv2.resize(segmented_image, (int(width), int(height)), interpolation = cv2.INTER_LINEAR) 

        row_current = 0
        object_ = {
            'x_start': -1,
            'x_end' : -1,
            'y_start' : -1,
            'y_end': -1
        }

        # Grayscale 
        #gray = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY) 
        
        # Find Canny edges 
        edged = cv2.Canny(segmented_image, 30, 200) 
        cv2.waitKey(0) 
        
        # Finding Contours 
        # Use a copy of the image e.g. edged.copy() 
        # since findContours alters the image 
        contours, hierarchy = cv2.findContours(edged,  
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 

        print("Number of Contours found = " + str(len(contours))) 

        object_matrix = []

        e_w = int(width / 100)
        e_h = int(height / 100)

        for contour in contours:
        

            extLeft = tuple(contour[contour[:, :, 0].argmin()][0])
            extRight = tuple(contour[contour[:, :, 0].argmax()][0])
            extTop = tuple(contour[contour[:, :, 1].argmin()][0])
            extBot = tuple(contour[contour[:, :, 1].argmax()][0])
            if (extRight[0] - extLeft[0] > 5 * e_w ) and (extBot[1] - extTop[1] > 5 * e_h):
                object_list = [(max(extLeft[0] - e_w, 0), min(extRight[0] + e_w , width)),(max(extTop[1] - e_h, 0), min(extBot[1] + e_h, height))]
                #object_list = [(extLeft[0],extRight[0]), (extTop[1], extBot[1])]
                object_matrix.append(object_list)

        areas = []
        areas_copy = []
        obj_matrix_largest = []
        for i in object_matrix :
            area = (i[0][1] - i[0][0]) * (i[1][1] - i[1][0])
            areas.append(area)
            areas_copy.append(area)

        
        if len(object_matrix) >= 2 :
            max_area = areas.index(max(areas))
            sec_max_area = areas.index(sorted(areas_copy, reverse=True)[1])
            obj_matrix_largest.append(object_matrix[max_area])
            obj_matrix_largest.append(object_matrix[sec_max_area])

        elif len(object_matrix) == 1 : 
            max_area = areas.index(max(areas))
            obj_matrix_largest.append(object_matrix[max_area])

        if len(obj_matrix_largest) > 0 :
            for i in obj_matrix_largest:
                original_image = cv2.rectangle(original_image, (i[0][0], i[1][0]), (i[0][1], i[1][1]), (0, 0, 0), 3) 


        return original_image




classes_dict = {'background': 0,'hand': 1}
classes_to_train = ['hand']

    
segmented = Segementation(classes_dict, classes_to_train, "Unet", "efficientnetb3", "seg_hand_trained/best_model.h5")

segmented.segment_video(VideoPath)