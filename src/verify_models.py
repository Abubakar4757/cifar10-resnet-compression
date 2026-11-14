"""Audit all final checkpoints, physical dimensions, sparsity and ONNX graphs."""
import argparse
import hashlib
from pathlib import Path
import numpy as np
import onnx
import torch
from engine import initialize,load_model,model_files,sparsity,write_json,write_csv
from benchmark import onnx_file,session


def verify(root,out):
    initialize();torch.set_num_threads(1)
    root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    x=np.random.default_rng(450).normal(size=(2,3,32,32)).astype(np.float32)
    rows=[];layers=[]
    for name,path in model_files(root).items():
        m=load_model(path)
        with torch.inference_mode():y=m(torch.from_numpy(x)).numpy()
        assert y.shape==(2,10) and np.isfinite(y).all()
        graph=onnx.load(str(onnx_file(root,name)));onnx.checker.check_model(graph)
        z=session(onnx_file(root,name)).run(None,{'images':x})[0]
        assert z.shape==y.shape and np.isfinite(z).all()
        int8=sum(v.data_type==onnx.TensorProto.INT8 for v in graph.graph.initializer)
        qdq=sum(n.op_type in ('QuantizeLinear','DequantizeLinear') for n in graph.graph.node)
        if name in ('ptq','qat'):
            assert int8>=21 and qdq>=40
            assert 'quantized::conv2d' in str(m.inlined_graph)
        else:
            assert not any('mask' in k or 'weight_orig' in k for k in m.state_dict())
            if name=='unstructured':assert .699<sparsity(m)['sparsity']<.701
            macs=0;handles=[]
            def hook(label):
                def record(layer,inp,output):
                    nonlocal macs
                    if isinstance(layer,torch.nn.Conv2d):
                        ops=output.shape[2]*output.shape[3]*layer.out_channels*layer.in_channels//layer.groups*layer.kernel_size[0]*layer.kernel_size[1]
                    else:ops=layer.in_features*layer.out_features
                    macs+=ops
                    layers.append({'model':name,'layer':label,'weight_shape':str(list(layer.weight.shape)),
                                   'weights':layer.weight.numel(),'MACs_per_image':int(ops),
                                   'output_shape':str(list(output.shape))})
                return record
            for label,layer in m.named_modules():
                if isinstance(layer,(torch.nn.Conv2d,torch.nn.Linear)):
                    handles.append(layer.register_forward_hook(hook(label)))
            with torch.inference_mode():m(torch.from_numpy(x[:1]))
            for h in handles:h.remove()
            if name=='structured':
                assert m.hidden==[40,40,80,80,160,160,320,320]
                for b,k in zip(m.blocks,m.hidden):
                    assert b.conv1.out_channels==b.bn1.num_features==b.conv2.in_channels==k
            write_json(out/f'{name}_architecture.json',{'MACs_per_image':macs,**sparsity(m),'hidden':m.hidden})
        rows.append({'model':name,'native_file':path.name,'native_bytes':path.stat().st_size,
                     'native_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                     'onnx_file':onnx_file(root,name).name,'onnx_bytes':onnx_file(root,name).stat().st_size,
                     'onnx_sha256':hashlib.sha256(onnx_file(root,name).read_bytes()).hexdigest(),
                     'int8_initializers':int8,'quantize_dequantize_nodes':qdq,
                     'output_shape':list(y.shape),'finite_outputs':True,
                     'random_input_max_logit_difference':float(np.abs(y-z).max())})
    write_json(out/'model_verification.json',rows);write_csv(out/'architecture_layers.csv',layers)
    print('VERIFIED FIVE CHECKPOINTS AND FIVE ONNX MODELS',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--models',default='artifacts')
    p.add_argument('--output',default='results');a=p.parse_args();verify(a.models,a.output)
