"""Training, PTQ, QAT and pruning implementations for Assignment 3."""
import copy
import csv
import hashlib
import json
import os
import platform
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn.utils import prune
from models import ResNet18, load_fp32, save_fp32, structured_prune
from data import Data, accuracy, SEED


def write_json(path,obj):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(obj,indent=2),encoding='utf-8')


def write_csv(path,rows):
    if not rows:return
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def initialize(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(2)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark=True
        torch.backends.cuda.matmul.allow_tf32=True
    engines=list(torch.backends.quantized.supported_engines)
    available=next((v for v in ('fbgemm','x86','onednn','qnnpack') if v in engines),None)
    if available is None:raise RuntimeError(f'No supported INT8 CPU backend: {engines}')
    torch.backends.quantized.engine=available
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def train(model,data,out,name,epochs,lr=.1,qat=False,masks=False):
    device=data.device
    model=model.to(device).to(memory_format=torch.channels_last)
    optimizer=torch.optim.SGD(model.parameters(),lr=lr,momentum=.9,weight_decay=5e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,epochs)
    amp=device.type=='cuda' and not qat
    scaler=torch.amp.GradScaler('cuda',enabled=amp)
    best=-1; best_state=None; rows=[]
    for epoch in range(epochs):
        start=time.perf_counter();model.train()
        if qat and epoch>=2:
            model.apply(torch.ao.nn.intrinsic.qat.freeze_bn_stats)
        if qat and epoch>=3:model.apply(torch.ao.quantization.disable_observer)
        total=correct=0; loss_sum=0
        batch=256 if device.type=='cuda' else 64
        for x,y in data.batches('train',batch,augment=True):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=amp):
                z=model(x);loss=nn.functional.cross_entropy(z,y)
            scaler.scale(loss).backward();scaler.step(optimizer);scaler.update()
            loss_sum+=float(loss.detach())*len(y)
            correct+=int((z.detach().argmax(1)==y).sum());total+=len(y)
        val=accuracy(model,data)
        if device.type=='cuda':torch.cuda.synchronize()
        row={'epoch':epoch+1,'loss':loss_sum/total,'train_accuracy':100*correct/total,
             'val_accuracy':val,'lr':optimizer.param_groups[0]['lr'],
             'seconds':time.perf_counter()-start}
        rows.append(row);print(name,json.dumps(row),flush=True)
        if val>best:
            best=val
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            torch.save({'state_dict':best_state,'hidden':model.hidden,'epoch':epoch+1,
                        'val_accuracy':best,'seed':SEED},Path(out)/f'{name}_best.pth')
        write_csv(Path(out)/f'{name}_history.csv',rows)
        scheduler.step()
    model.load_state_dict(best_state)
    return model.cpu().eval(),{'best_val_accuracy':best,'epochs':epochs,
                              'epoch_seconds':sum(r['seconds'] for r in rows)}


def run_baseline(args):
    device=initialize(); out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    data=Data(args.data,device,out)
    model,summary=train(ResNet18(),data,out,'baseline',args.epochs,.1)
    save_fp32(model,out/'baseline_fp32.pt',**summary,seed=SEED)
    write_json(out/'baseline_summary.json',summary)
    return model


def quant_config(qat=False):
    # UINT8 activations with a reduced observer range; INT8 per-channel weights.
    ao=torch.ao.quantization
    if qat:
        return ao.QConfig(
            activation=ao.FakeQuantize.with_args(observer=ao.MovingAverageMinMaxObserver,
                quant_min=0,quant_max=127,dtype=torch.quint8,qscheme=torch.per_tensor_affine),
            weight=ao.FakeQuantize.with_args(observer=ao.MovingAveragePerChannelMinMaxObserver,
                quant_min=-128,quant_max=127,dtype=torch.qint8,
                qscheme=torch.per_channel_symmetric,ch_axis=0))
    return ao.QConfig(
        activation=ao.HistogramObserver.with_args(bins=2048,dtype=torch.quint8,
            qscheme=torch.per_tensor_affine,reduce_range=True),
        weight=ao.PerChannelMinMaxObserver.with_args(dtype=torch.qint8,
            qscheme=torch.per_channel_symmetric,ch_axis=0))


def save_int8(model,path):
    model.eval().cpu()
    sample=torch.randn(1,3,32,32)
    with torch.inference_mode(): traced=torch.jit.trace(model,sample)
    torch.jit.save(traced,str(path))
    reloaded=torch.jit.load(str(path)).eval()
    torch.testing.assert_close(model(sample),reloaded(sample),rtol=0,atol=0)


def sparsity(model):
    weights=[m.weight.detach() for m in model.modules() if isinstance(m,(nn.Conv2d,nn.Linear))]
    total=sum(w.numel() for w in weights)
    zeros=sum(int((w==0).sum()) for w in weights)
    return {'weight_count':total,'zero_weights':zeros,'sparsity':zeros/total,
            'parameters':sum(p.numel() for p in model.parameters())}


def load_model(path):
    path=Path(path)
    if 'int8' in path.name:return torch.jit.load(str(path),map_location='cpu').eval()
    return load_fp32(path)[0]


def model_files(root):
    root=Path(root)
    return {'fp32':root/'baseline_fp32.pt','ptq':root/'ptq_int8.pt',
            'qat':root/'qat_int8.pt','unstructured':root/'unstructured_fp32.pt',
            'structured':root/'structured_fp32.pt'}
