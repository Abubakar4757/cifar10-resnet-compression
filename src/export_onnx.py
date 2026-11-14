"""ONNX deployment, including QDQ graphs using PyTorch's trained qparameters.

The quantized export does not re-calibrate or silently substitute an FP32 model.
It unpacks the converted PyTorch INT8 tensors and preserves their output scales.
ONNX Runtime can fuse supported QDQ patterns into CPU integer kernels.
"""
import argparse
from pathlib import Path
import numpy as np
import torch
import onnx
from onnx import helper as h, numpy_helper as nh, TensorProto as T


def export_quantized(model,path):
    nodes=[];initializers=[];serial=0
    def init(name,value):
        initializers.append(nh.from_array(np.asarray(value),name));return name
    def op(kind,inputs,label,**attrs):
        nonlocal serial
        serial+=1;out=f'{label}_{serial}'
        nodes.append(h.make_node(kind,inputs,[out],name=out,**attrs));return out
    def params(label,scale,zp):
        return (init(label+'_scale',np.asarray(float(scale),np.float32)),
                init(label+'_zero',np.asarray(int(zp),np.uint8)))
    def qdq(x,p,label):
        q=op('QuantizeLinear',[x,*p],label+'_q')
        return op('DequantizeLinear',[q,*p],label+'_dq')
    def conv(x,layer,label,relu=False):
        w=layer.weight()
        wi=init(label+'_weight_int8',w.int_repr().cpu().numpy())
        ws=init(label+'_weight_scale',w.q_per_channel_scales().float().cpu().numpy())
        wz=init(label+'_weight_zero',w.q_per_channel_zero_points().to(torch.int8).cpu().numpy())
        wd=op('DequantizeLinear',[wi,ws,wz],label+'_wdq',axis=0)
        bias=init(label+'_bias',layer.bias().detach().cpu().numpy() if layer.bias() is not None
                  else np.zeros(w.shape[0],np.float32))
        pad=list(layer.padding);stride=list(layer.stride);dil=list(layer.dilation)
        y=op('Conv',[x,wd,bias],label,pads=pad+pad,strides=stride,dilations=dil,group=layer.groups)
        if relu:y=op('Relu',[y],label+'_relu')
        p=params(label,layer.scale,layer.zero_point)
        return qdq(y,p,label),p
    inp=params('input',model.quant.scale,model.quant.zero_point)
    x=qdq('images',inp,'input')
    x,p=conv(x,model.conv,'stem',True)
    for i,b in enumerate(model.blocks):
        shortcut=x
        y,_=conv(x,b.conv1,f'b{i}_conv1',True)
        y,_=conv(y,b.conv2,f'b{i}_conv2')
        if not isinstance(b.skip,torch.nn.Identity):shortcut,_=conv(shortcut,b.skip[0],f'b{i}_skip')
        x=op('Add',[y,shortcut],f'b{i}_add')
        x=op('Relu',[x],f'b{i}_relu')
        p=params(f'b{i}_add_out',b.add.scale,b.add.zero_point)
        x=qdq(x,p,f'b{i}_out')
    x=op('GlobalAveragePool',[x],'pool')
    x=qdq(x,p,'pool_round')
    x=op('Flatten',[x],'flatten',axis=1)
    fc=model.fc;w=fc.weight()
    wi=init('fc_weight_int8',w.int_repr().cpu().numpy())
    ws=init('fc_weight_scale',w.q_per_channel_scales().float().cpu().numpy())
    wz=init('fc_weight_zero',w.q_per_channel_zero_points().to(torch.int8).cpu().numpy())
    wd=op('DequantizeLinear',[wi,ws,wz],'fc_wdq',axis=0)
    bias=init('fc_bias',fc.bias().detach().cpu().numpy())
    x=op('Gemm',[x,wd,bias],'fc',transB=1)
    p=params('fc_output',fc.scale,fc.zero_point)
    x=qdq(x,p,'fc_output')
    nodes.append(h.make_node('Identity',[x],['logits']))
    graph=h.make_graph(nodes,'CIFAR ResNet18 PyTorch INT8 QDQ',
        [h.make_tensor_value_info('images',T.FLOAT,['batch',3,32,32])],
        [h.make_tensor_value_info('logits',T.FLOAT,['batch',10])],initializers)
    model_onnx=h.make_model(graph,opset_imports=[h.make_opsetid('',17)],producer_name='Assignment3')
    model_onnx.ir_version=9
    onnx.checker.check_model(model_onnx)
    onnx.save(model_onnx,str(path))


def export_float(model,path):
    model.eval().cpu()
    with torch.inference_mode():
        torch.onnx.export(model,torch.randn(1,3,32,32),str(path),opset_version=17,
            input_names=['images'],output_names=['logits'],
            dynamic_axes={'images':{0:'batch'},'logits':{0:'batch'}},dynamo=False)
    onnx.checker.check_model(onnx.load(str(path)))


def export_all(root):
    from engine import model_files,load_model
    for name,path in model_files(root).items():
        if name in ('ptq','qat'):
            assert (Path(root)/f'{name}_int8.onnx').exists()
            continue
        export_float(load_model(path),Path(root)/f'{name}_fp32.onnx')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--models',default='artifacts')
    export_all(p.parse_args().models)
