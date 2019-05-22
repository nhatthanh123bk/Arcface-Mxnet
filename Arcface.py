import mxnet as mx
from sklearn import preprocessing
import numpy as np
import cv2
def create_model():
    ctx = mx.cpu()
    sym, arg_params, aux_params = mx.model.load_checkpoint('./Model/model-r100-ii/model', 0)
    all_layers = sym.get_internals()
    sym = all_layers['fc1' + '_output']
    mod = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
    mod.bind(for_training=False, data_shapes=[('data', (1,3,112,112))])
    mod.set_params(arg_params, aux_params)
    return mod
    
class ext_embedding(): 
    def __init__(self):
        self.mod = create_model()
        

    def ext_emb(self,img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img,axis=0)
        data = mx.nd.array(img)
        db = mx.io.DataBatch(data=(data,))
        self.mod.forward(db, is_train= False)
        embedding = self.mod.get_outputs()[0].asnumpy()
        embedding = preprocessing.normalize(embedding).flatten()
        return embedding