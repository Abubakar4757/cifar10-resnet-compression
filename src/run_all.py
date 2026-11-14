"""Reproduce Assignment 3 from training through deployment and benchmarking."""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',default='data')
    p.add_argument('--output',default='artifacts');p.add_argument('--results',default='results')
    p.add_argument('--baseline-epochs',type=int,default=50)
    p.add_argument('--qat-epochs',type=int,default=6)
    p.add_argument('--pruning-epochs',type=int,default=3)
    p.add_argument('--structured-epochs',type=int,default=8)
    p.add_argument('--skip-benchmark',action='store_true')
    p.add_argument('--resume',action='store_true',help='Skip completed stages with final checkpoints')
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    src=Path(__file__).resolve().parent
    commands=[('train.py','baseline_fp32.pt',['--epochs',str(a.baseline_epochs)]),
              ('ptq.py','ptq_int8.pt',[]),('qat.py','qat_int8.pt',['--epochs',str(a.qat_epochs)]),
              ('prune_unstructured.py','unstructured_fp32.pt',['--epochs',str(a.pruning_epochs)]),
              ('prune_structured.py','structured_fp32.pt',['--epochs',str(a.structured_epochs)])]
    stages=[]
    for script,checkpoint,extra in commands:
        complete=(out/checkpoint).exists()
        if script in ('ptq.py','qat.py'):
            complete=complete and (out/(script[:-3]+'_int8.onnx')).exists()
        if a.resume and complete:continue
        cmd=[sys.executable,str(src/script),'--data',a.data,'--output',a.output,*extra]
        if script!='train.py':cmd+=['--baseline',str(out/'baseline_fp32.pt')]
        start=time.perf_counter();print('RUN',cmd,flush=True);subprocess.run(cmd,check=True)
        stages.append({'script':script,'seconds':time.perf_counter()-start})
        (out/'stage_times.json').write_text(json.dumps(stages,indent=2))
    subprocess.run([sys.executable,str(src/'export_onnx.py'),'--models',a.output],check=True)
    if not a.skip_benchmark:
        subprocess.run([sys.executable,str(src/'benchmark.py'),'--models',a.output,
                        '--data',a.data,'--output',a.results],check=True)
    import torch,torchvision,onnx,onnxruntime
    env={'python':platform.python_version(),'torch':torch.__version__,
         'torchvision':torchvision.__version__,'onnx':onnx.__version__,'onnxruntime':onnxruntime.__version__,
         'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
         'seed':450,'arguments':vars(a)}
    (out/'training_environment.json').write_text(json.dumps(env,indent=2))
    manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()
              if p.suffix in ('.pt','.onnx')}
    (out/'model_sha256.json').write_text(json.dumps(manifest,indent=2))
    print('ALL STAGES COMPLETE',json.dumps(env),flush=True)


if __name__=='__main__':main()
