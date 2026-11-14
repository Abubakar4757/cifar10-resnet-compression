"""Independent checks of physical channel slicing and masked recovery gradients."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import copy
import torch
from torch.nn.utils import prune
from models import ResNet18,structured_prune
from engine import initialize

initialize();torch.set_num_threads(1)
m=ResNet18().eval();n,records=structured_prune(m,.375);n.eval()
masked=copy.deepcopy(m)
with torch.no_grad():
    for b,r in zip(masked.blocks,records):
        keep=torch.zeros(b.bn1.num_features,dtype=torch.bool)
        keep[r['kept_indices']]=True
        b.bn1.weight[~keep]=0;b.bn1.bias[~keep]=0
x=torch.randn(2,3,32,32)
with torch.inference_mode():
    torch.testing.assert_close(masked(x),n(x),rtol=1e-4,atol=2e-6)
assert sum(p.numel() for p in n.parameters())<sum(p.numel() for p in m.parameters())
layer=m.blocks[0].conv1
prune.l1_unstructured(layer,'weight',amount=.7)
m.train();optimizer=torch.optim.SGD(m.parameters(),lr=.01,momentum=.9)
before=layer.weight_mask.clone()
optimizer.zero_grad();m(x).square().mean().backward();optimizer.step()
assert torch.count_nonzero(layer.weight_orig.grad[before==0])==0
layer.weight=layer.weight_orig*layer.weight_mask
prune.remove(layer,'weight')
assert torch.count_nonzero(layer.weight[before==0])==0
assert not hasattr(layer,'weight_orig')
print('Physical slicing matches an explicitly gated network; masked gradients stay zero; permanence verified.')
