"""Iterative global magnitude pruning with masked recovery and permanence."""
from pathlib import Path
import torch
from torch import nn
from torch.nn.utils import prune
from cli import parser
from models import load_fp32
from data import Data, accuracy
from engine import initialize, train, write_json, sparsity


def run_unstructured(args):
    device=initialize();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    m,_=load_fp32(args.baseline);data=Data(args.data,device)
    targets=[.3,.5,.7]; iterations=[]
    for i,target in enumerate(targets):
        m=m.to(memory_format=torch.contiguous_format)
        for layer in m.modules():
            if hasattr(layer,'weight_orig'):
                layer.weight=layer.weight_orig*layer.weight_mask
        # Use actual permanent masks between rounds; masked gradients stay zero
        # throughout recovery fine-tuning and therefore cannot regrow pruned weights.
        pairs=[(v,'weight') for v in m.modules() if isinstance(v,(nn.Conv2d,nn.Linear))]
        current=sparsity(m)['sparsity']
        prune.global_unstructured(pairs,pruning_method=prune.L1Unstructured,
                                  amount=(target-current)/(1-current))
        before=accuracy(m.to(device),data)
        m,info=train(m,data,out,f'unstructured_round_{i+1}',args.epochs,.01,masks=True)
        for layer,name in pairs:
            layer.weight=layer.weight_orig*layer.weight_mask
        after=accuracy(m.to(device),data);m.cpu()
        iterations.append({'round':i+1,'target_sparsity':target,'before_finetune_val':before,
                           'after_finetune_val':after,**sparsity(m),**info})
        print('PRUNING',iterations[-1],flush=True)
    for layer,name in pairs:prune.remove(layer,name)
    # Save an ordinary checkpoint for the repository submission.
    payload={'state_dict':m.state_dict(),'hidden':m.hidden,'architecture':'CIFAR ResNet18',
             'pruning':'global iterative L1 magnitude','iterations':iterations}
    torch.save(payload,out/'unstructured_fp32.pt')
    write_json(out/'unstructured_summary.json',iterations)
    return m


if __name__=='__main__':run_unstructured(parser('unstructured').parse_args())
