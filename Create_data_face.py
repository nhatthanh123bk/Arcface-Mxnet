import sys
sys.path.insert(0,"/home/nhatthanh/Desktop/Recoginize_Idol/mtcnn")
from mtcnn_detector import MtcnnDetector
from face_preprocess import preprocess 
import mxnet as mx
import cv2
import numpy as np
import os

path1 = "/home/nhatthanh/Desktop/Recoginize_Idol/data/images/"
path2 = "/home/nhatthanh/Desktop/Recoginize_Idol/data/images_croped/"

def main():
    detector = MtcnnDetector(model_folder='./mtcnn/model', ctx=mx.cpu(), num_worker = 4 , accurate_landmark = True)
    for i in os.listdir(path1):
        if(((i in os.listdir(path2)) == True)):
            continue
        os.mkdir(path2+str(i))
        for j in os.listdir(path1+str(i)):
            img = cv2.imread(path1 + str(i)+"/"+j,1)
            results= detector.detect_face(img)
            if results is None:
                continue    
            if results is not None:
                total_boxes = results[0]
                points = results[1]
                for id in range(len(points)):
                    point = points[id].reshape((2, 5)).T
                    nimg = preprocess(img, total_boxes[id], point, image_size='112,112')
                    cv2.imwrite(path2+str(i)+"/"+str(j)+"_"+str(id)+"jpg",nimg)
           
if __name__ == '__main__':
    main()
