import numpy as np 
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
def main():
    embs_train = np.load("./embs/embs_train.npy")
    label_train = np.load("./embs/label_train.npy")
    print(label_train)
    X_train, X_test, Y_train, Y_test = train_test_split(embs_train,label_train,test_size = 0.1)
    print(Y_test.shape[0])
    model_svm = SVC(kernel='linear', probability=True)
    model_svm.fit(X_train,Y_train)
    Y_pred = model_svm.predict(X_test)    
    print(accuracy_score(Y_pred,Y_test))
    filename = "./Model/model-classifer/model_svm.sav"
    pickle.dump((model_svm),open(filename,"wb"))
    print("model is saved...")

if __name__ == "__main__":
    main()    