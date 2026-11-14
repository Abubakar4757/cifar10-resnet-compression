"""Post-training static quantization with fusion and training-only calibration."""
from pathlib import Path
import time
import numpy as np
import torch
from cli import parser
from models import load_fp32
from data import Data, accuracy, SEED
from engine import initialize, quant_config, save_int8, write_json


def run_ptq(args):
    initialize();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    m,_=load_fp32(args.baseline);m.eval();m.fuse()
    m.qconfig=quant_config()
    torch.ao.quantization.prepare(m,inplace=True)
    data=Data(args.data,'cpu')
    generator=torch.Generator().manual_seed(SEED)
    order=torch.randperm(len(data.arrays['train'][1]),generator=generator)
    images,labels=data.arrays['train']
    data.arrays['train']=(images[order],labels[order])
    calibration_indices=data.train_ids[order[:args.calibration].numpy()]
    np.save(out/'calibration_indices.npy',calibration_indices)
    start=time.perf_counter();seen=0
    with torch.inference_mode():
        for x,y in data.batches('train',64,limit=args.calibration):m(x);seen+=len(y)
    m=torch.ao.quantization.convert(m,inplace=False).eval()
    save_int8(m,out/'ptq_int8.pt')
    from export_onnx import export_quantized
    export_quantized(m,out/'ptq_int8.onnx')
    summary={'calibration_images':seen,'calibration_source':'training split only',
             'observer':'HistogramObserver 2048 bins','activation':'uint8 affine per-tensor reduced observer range 0..127',
             'weights':'int8 symmetric per-output-channel','backend':torch.backends.quantized.engine,
             'calibration_class_counts':torch.bincount(labels[order[:args.calibration]],minlength=10).tolist(),
             'seconds':time.perf_counter()-start,'val_accuracy':accuracy(m,data)}
    write_json(out/'ptq_summary.json',summary);print('PTQ',summary,flush=True)
    return m


if __name__=='__main__':run_ptq(parser('ptq').parse_args())
