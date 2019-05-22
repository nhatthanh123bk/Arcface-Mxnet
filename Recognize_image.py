import sys
sys.path.insert(0,"/home/nhatthanh/Desktop/Recoginize_Idol/mtcnn")
from mtcnn_detector import MtcnnDetector
from face_preprocess import preprocess 
from Arcface import ext_embedding
import mxnet as mx
import cv2
import numpy as np
import os
import argparse
import pickle


labels = {1:"Bang_Kieu",2:"Dam_Vinh_Hung",3:"Ha_Anh_Tuan",4:"Ho_Ngoc_Ha",5:"My_Linh",6:"My_Tam",7:"Noo_Phuoc_Thinh",8:"Son_Tung",9:"Toc_Tien",10:"Tuan_Hung",11:"Tung_Duong",12:"Ung_Hoang_Phuc",13:"Dong_Nhi",14:"Soobin_HS",15:"Issac"}

detector = MtcnnDetector(model_folder='./mtcnn/model', ctx=mx.cpu(), num_worker = 4 , accurate_landmark = True)
ext_embedding = ext_embedding()
def create_emb(image):
	embs = []
	results = detector.detect_face(image)
	if results is not None:
		total_boxes = results[0]
		points = results[1]
		for id in range(len(points)):
			point = points[id].reshape((2, 5)).T
			nimg = preprocess(image, total_boxes[id], point, image_size='112,112')
			emb = ext_embedding.ext_emb(nimg)
			embs.append(emb)
	return total_boxes,np.asarray(embs)       



def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--image",help = "path image!!!")
	args = vars(ap.parse_args())
	img = cv2.imread(args["image"])

	coordinates,embs = create_emb(img)
	model_predict = pickle.load(open("./Model/model-classifer/model_svm.sav","rb"))
	label = model_predict.predict(embs)

	for i in range(embs.shape[0]):
		bb = np.zeros(4)
		bb[0] = coordinates[i][0]
		bb[1] = coordinates[i][1]
		bb[2] = coordinates[i][2]
		bb[3] = coordinates[i][3]
		cv2.rectangle(img, (int(bb[0]),int(bb[1])),(int(bb[2]),int(bb[3])), (0, 255, 0), 1)
		startX = bb[0]
		startY = bb[1] - 15 if bb[1] - 15 > 15 else bb[1] + 15
		print(labels[label[i]])
		cv2.putText(img,labels[label[i]], (int(startX), int(startY)),cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
	print(img.shape)
	cv2.imshow("image",img)
	cv2.waitKey(0)    

if __name__ == '__main__':
		main()	
