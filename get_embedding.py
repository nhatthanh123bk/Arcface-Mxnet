import cv2
import numpy as np
import os
from Arcface import ext_embedding

path_data = "/home/nhatthanh/Desktop/Recoginize_Idol/data/images_croped/"
ext_embedding = ext_embedding()

def get_embedding_train(path):
    embs_train = []
    label_train = []
    for i in os.listdir(path):
        for j in os.listdir(path+i):
            img = cv2.imread(path+i+"/"+j,1)
            emb_train = ext_embedding.ext_emb(img)
            embs_train.append(emb_train)
            label_train.append(int(i))
    return np.asarray(embs_train),np.asarray(label_train)        


def main():
    embs_train,label_train = get_embedding_train(path_data)
    np.save("./embs/embs_train.npy",embs_train)
    np.save("./embs/label_train.npy",label_train)

if __name__== "__main__":
    main()