"""Evaluate all five checkpoints and ONNX models on all 10,000 test images.

Accuracy uses batching. Separate warm batch-one trials measure inference only,
excluding preprocessing, data loading, and session construction.
"""
import argparse
import gc
import hashlib
import json
import os
import platform
import time
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
import psutil
import torch
from data import Data
from engine import initialize,model_files,load_model,write_csv,write_json,sparsity


def session(path):
    options=ort.SessionOptions()
    options.intra_op_num_threads=1;options.inter_op_num_threads=1
    options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
    options.add_session_config_entry('session.intra_op.allow_spinning','0')
    options.add_session_config_entry('session.inter_op.allow_spinning','0')
    return ort.InferenceSession(str(path),options,providers=['CPUExecutionProvider'])


def onnx_file(root,name):
    return Path(root)/(f'{name}_int8.onnx' if name in ('ptq','qat') else f'{name}_fp32.onnx')


def benchmark(args):
    initialize();torch.set_num_threads(1)
    try:torch.set_num_interop_threads(1)
    except RuntimeError:pass
    root=Path(args.models);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    data=Data(args.data,'cpu')
    # Prepare outside timing, retaining batch-one FP32 NCHW input.
    bx,by=next(data.batches('test',128))
    inputs=bx.contiguous().numpy()
    predictions=[];summary=[];samples=[];parity=[];reference={}
    process=psutil.Process()
    for runtime in ['pytorch','onnxruntime']:
        for name,path in model_files(root).items():
            gc.collect();before=process.memory_info().rss
            if runtime=='pytorch':
                model=load_model(path).eval()
                def infer(x):
                    with torch.inference_mode():return model(torch.from_numpy(x)).numpy()
                saved=path
            else:
                saved=onnx_file(root,name);model=session(saved)
                def infer(x):return model.run(None,{'images':x})[0]
            after=process.memory_info().rss
            all_z=[];index=0;correct=0;eval_start=time.perf_counter()
            for x,y in data.batches('test',args.batch):
                arr=np.ascontiguousarray(x.numpy())
                start=time.perf_counter_ns();z=infer(arr);elapsed=(time.perf_counter_ns()-start)/1e6
                pred=z.argmax(1);truth=y.numpy();correct+=int((pred==truth).sum())
                prob=np.exp(z-z.max(1,keepdims=True));prob/=prob.sum(1,keepdims=True)
                for k in range(len(y)):
                    predictions.append({'runtime':runtime,'model':name,'test_index':index+k,
                        'true_label':int(truth[k]),'predicted_label':int(pred[k]),
                        'confidence':float(prob[k,pred[k]]),'correct':int(pred[k]==truth[k]),
                        'evaluation_batch_amortized_ms':elapsed/len(y)})
                all_z.append(z);index+=len(y)
            eval_seconds=time.perf_counter()-eval_start
            logits=np.concatenate(all_z)
            if runtime=='pytorch':reference[name]=logits
            else:
                ref=reference[name];difference=logits-ref
                row={'model':name,'test_images':index,
                     'top1_agreement_percent':100*float((logits.argmax(1)==ref.argmax(1)).mean()),
                     'max_absolute_logit_error':float(np.abs(difference).max()),
                     'mean_absolute_logit_error':float(np.abs(difference).mean()),
                     'accuracy_difference_pp':100*float(((logits.argmax(1)==data.arrays['test'][1].numpy()).mean())-
                         ((ref.argmax(1)==data.arrays['test'][1].numpy()).mean()))}
                parity.append(row)
                # Floating exports should agree closely; quantized integer
                # rounding and kernel implementations can differ by a few LSBs.
                if name not in ('ptq','qat'):
                    assert row['max_absolute_logit_error']<.005,row
                else:assert row['top1_agreement_percent']>=98,row
            times=[];wall=0
            for repeat in range(args.repeats):
                for i in range(args.warmup):infer(inputs[i%len(inputs):i%len(inputs)+1])
                start_wall=time.perf_counter()
                for i in range(args.samples):
                    arr=inputs[i%len(inputs):i%len(inputs)+1]
                    start=time.perf_counter_ns();infer(arr);ms=(time.perf_counter_ns()-start)/1e6
                    times.append(ms)
                    samples.append({'runtime':runtime,'model':name,'repeat':repeat+1,
                                    'sample':i,'latency_ms':ms})
                wall+=time.perf_counter()-start_wall
            stats={'runtime':runtime,'model':name,'test_images':index,
                   'test_accuracy_percent':100*correct/index,'model_file_bytes':saved.stat().st_size,
                   'batch_one_mean_ms':float(np.mean(times)),'batch_one_p50_ms':float(np.percentile(times,50)),
                   'batch_one_p99_ms':float(np.percentile(times,99)),
                   'batch_one_throughput_images_s':len(times)/wall,
                   'warmups_per_repeat':args.warmup,'measured_samples':len(times),
                   'threads':1,'evaluation_batch_size':args.batch,'evaluation_seconds':eval_seconds,
                   'process_rss_after_load_mib':after/2**20,
                   'rss_load_change_mib':(after-before)/2**20}
            summary.append(stats);print('BENCHMARK',json.dumps(stats),flush=True)
            write_csv(out/'benchmark_summary.csv',summary)
            write_csv(out/'onnx_parity.csv',parity)
            del model
    write_csv(out/'test_predictions.csv',predictions)
    write_csv(out/'latency_samples.csv',samples)
    for name in ('fp32','unstructured','structured'):
        m=load_model(model_files(root)[name])
        write_json(out/f'{name}_parameter_stats.json',sparsity(m))
    env={'python':platform.python_version(),'os':platform.platform(),'processor':platform.processor(),
         'physical_cores':psutil.cpu_count(logical=False),'logical_cores':psutil.cpu_count(),
         'ram_gib':psutil.virtual_memory().total/2**30,
         'torch':torch.__version__,'onnx':onnx.__version__,'onnxruntime':ort.__version__,
         'providers':['CPUExecutionProvider'],'backend':torch.backends.quantized.engine,
         'timing':'one thread, batch one, no preprocessing, idle system, no contention workers'}
    write_json(out/'benchmark_environment.json',env)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--models',default='artifacts')
    p.add_argument('--data',default='data');p.add_argument('--output',default='results')
    p.add_argument('--batch',type=int,default=64);p.add_argument('--warmup',type=int,default=30)
    p.add_argument('--samples',type=int,default=300);p.add_argument('--repeats',type=int,default=3)
    benchmark(p.parse_args())
