"""Fixed stratified splits and GPU-resident CIFAR batches, without test leakage."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torchvision.datasets import CIFAR10

MEAN = (.4914,.4822,.4465)
STD = (.2470,.2435,.2616)
SEED = 450


class Data:
    def __init__(self, root, device='cpu', output=None):
        train = CIFAR10(root, train=True, download=True)
        test = CIFAR10(root, train=False, download=True)
        y = np.array(train.targets)
        rng = np.random.default_rng(SEED)
        ti,vi = [],[]
        for c in range(10):
            ids = rng.permutation(np.flatnonzero(y==c))
            vi.extend(ids[:500]); ti.extend(ids[500:])
        self.train_ids = np.array(ti); self.val_ids = np.array(vi)
        self.device = torch.device(device)
        self.arrays = {}
        for name,xx,yy in [('train',train.data[ti],y[ti]),
                            ('val',train.data[vi],y[vi]),
                            ('test',test.data,np.array(test.targets))]:
            self.arrays[name] = (torch.tensor(xx.transpose(0,3,1,2).copy(),
                                              device=self.device),
                                  torch.tensor(yy,device=self.device,dtype=torch.long))
        self.mean = torch.tensor(MEAN,device=self.device)[None,:,None,None]
        self.std = torch.tensor(STD,device=self.device)[None,:,None,None]
        if output:
            Path(output).mkdir(parents=True,exist_ok=True)
            np.savez_compressed(Path(output)/'split_indices.npz',train=ti,val=vi)

    def batches(self, split, batch=128, augment=False, limit=None):
        images,targets = self.arrays[split]
        count = len(targets) if limit is None else min(limit,len(targets))
        ids = (torch.randperm(len(targets),device=self.device)[:count] if augment
               else torch.arange(count,device=self.device))
        for start in range(0,count,batch):
            ix = ids[start:start+batch]
            x = images[ix].float()/255
            if augment:
                x = F.pad(x,(4,4,4,4),mode='constant',value=0)
                n = len(ix)
                offsets = torch.randint(0,9,(n,2),device=self.device)
                rows = torch.arange(32,device=self.device)[None,:,None] + offsets[:,0,None,None]
                cols = torch.arange(32,device=self.device)[None,None,:] + offsets[:,1,None,None]
                x = x.permute(0,2,3,1)[torch.arange(n,device=self.device)[:,None,None],rows,cols].permute(0,3,1,2)
                flip = torch.rand(n,device=self.device)<.5
                x[flip] = x[flip].flip(-1)
            yield ((x-self.mean)/self.std).contiguous(memory_format=torch.channels_last),targets[ix]


@torch.inference_mode()
def accuracy(model,data,split='val',batch=256):
    model.eval(); correct=total=0
    dev = next(model.parameters(),torch.empty(0)).device
    for x,y in data.batches(split,batch):
        pred=model(x.to(dev)).argmax(1)
        correct+=int((pred.cpu()==y.cpu()).sum()); total+=len(y)
    return 100*correct/total
