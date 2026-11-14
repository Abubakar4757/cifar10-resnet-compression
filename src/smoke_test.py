"""Export compatibility tests on random inputs; not experimental results."""
import copy
import json
from pathlib import Path
import numpy as np
import onnxruntime as ort
import torch
from models import ResNet18,structured_prune
from engine import initialize,quant_config,save_int8,write_json
from export_onnx import export_quantized,export_float


def smoke(out='smoke'):
    initialize();out=Path(out);out.mkdir(parents=True,exist_ok=True)
    m=ResNet18().eval();x=torch.randn(4,3,32,32)
    records=[]
    for mode in ('fp32','ptq','qat','structured'):
        n=copy.deepcopy(m)
        if mode=='structured':n,_=structured_prune(n,.375)
        if mode in ('ptq','qat'):
            n.train(mode=='qat');n.fuse(mode=='qat');n.qconfig=quant_config(mode=='qat')
            if mode=='qat':
                torch.ao.quantization.prepare_qat(n,inplace=True)
                n(x).sum().backward()
            else:
                torch.ao.quantization.prepare(n,inplace=True)
                with torch.no_grad():n(x)
            n.eval();n=torch.ao.quantization.convert(n,inplace=False)
            export_quantized(n,out/f'{mode}.onnx');save_int8(n,out/f'{mode}.pt')
        else:export_float(n,out/f'{mode}.onnx')
        opts=ort.SessionOptions();opts.intra_op_num_threads=1;opts.inter_op_num_threads=1
        session=ort.InferenceSession(str(out/f'{mode}.onnx'),opts,providers=['CPUExecutionProvider'])
        with torch.inference_mode():ref=n(x).numpy()
        actual=session.run(None,{'images':x.numpy()})[0]
        err=float(np.abs(actual-ref).max())
        scale=float(n.fc.scale) if mode in ('ptq','qat') else 1
        assert err<max(.001,scale*8),(mode,err,scale)
        records.append({'mode':mode,'max_absolute_error':err,'output_scale':scale,
                        'native_shape':list(ref.shape),'onnx_shape':list(actual.shape)})
    write_json(out/'smoke.json',records);print(json.dumps(records,indent=2),flush=True)

if __name__=='__main__':smoke()
