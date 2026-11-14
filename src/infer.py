"""CPU image inference for all checkpoint variants; input PIL RGB image."""
import argparse
import json
import numpy as np
import torch
from PIL import Image
from data import MEAN,STD
from engine import initialize,load_model

CLASSES=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']


def preprocess(path):
    image=Image.open(path).convert('RGB').resize((32,32),Image.Resampling.BILINEAR)
    x=np.asarray(image,dtype=np.float32).transpose(2,0,1)/255
    return np.ascontiguousarray(((x-np.array(MEAN,np.float32)[:,None,None])/
                                np.array(STD,np.float32)[:,None,None])[None])


def main(onnx_only=False):
    p=argparse.ArgumentParser();p.add_argument('--model',required=True)
    p.add_argument('--image',required=True);a=p.parse_args()
    initialize();torch.set_num_threads(1);x=preprocess(a.image)
    if a.model.endswith('.onnx') or onnx_only:
        from benchmark import session
        z=session(a.model).run(None,{'images':x})[0][0]
    else:
        with torch.inference_mode():z=load_model(a.model)(torch.from_numpy(x))[0].numpy()
    prob=np.exp(z-z.max());prob/=prob.sum();idx=int(prob.argmax())
    print(json.dumps({'class_index':idx,'class_name':CLASSES[idx],'confidence':float(prob[idx])},indent=2))


if __name__=='__main__':main()
