"""CIFAR-10 ResNet-18 with explicit quantizable residual additions.

The 3x3 stride-one stem and omitted max-pool retain CIFAR spatial detail.
There are two BasicBlocks per stage, as in ResNet-18. Structured pruning
reduces the internal width of each block, keeping the residual interface
unchanged; conv2 input columns are sliced with the same channel indices.
"""
import torch
from torch import nn
from torch.ao.quantization import QuantStub, DeQuantStub


class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride=1, hidden=None):
        super().__init__()
        hidden = cout if hidden is None else hidden
        self.conv1 = nn.Conv2d(cin, hidden, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(hidden, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.skip = (nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False),
                                   nn.BatchNorm2d(cout))
                     if stride != 1 or cin != cout else nn.Identity())
        self.add = torch.ao.nn.quantized.FloatFunctional()

    def forward(self, x):
        y = self.relu1(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return self.add.add_relu(y, self.skip(x))


class ResNet18(nn.Module):
    def __init__(self, hidden=None):
        super().__init__()
        self.hidden = list(hidden or [64,64,128,128,256,256,512,512])
        self.quant = QuantStub()
        self.conv = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=False)
        blocks, cin = [], 64
        for i, cout in enumerate([64,64,128,128,256,256,512,512]):
            stride = 2 if i in (2,4,6) else 1
            blocks.append(BasicBlock(cin, cout, stride, self.hidden[i]))
            cin = cout
        self.blocks = nn.ModuleList(blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, 10)
        self.dequant = DeQuantStub()
        for m in self.modules():
            if isinstance(m, nn.Conv2d): nn.init.kaiming_normal_(m.weight, mode='fan_out')

    def forward(self, x):
        x = self.relu(self.bn(self.conv(self.quant(x))))
        for block in self.blocks: x = block(x)
        return self.dequant(self.fc(torch.flatten(self.pool(x), 1)))

    def fuse(self, qat=False):
        f = (torch.ao.quantization.fuse_modules_qat if qat
             else torch.ao.quantization.fuse_modules)
        f(self, [['conv','bn','relu']], inplace=True)
        for b in self.blocks:
            f(b, [['conv1','bn1','relu1'],['conv2','bn2']], inplace=True)
            if isinstance(b.skip, nn.Sequential): f(b.skip, [['0','1']], inplace=True)


def save_fp32(model, path, **metadata):
    torch.save({'state_dict': model.cpu().state_dict(), 'hidden': model.hidden,
                'architecture': 'CIFAR ResNet18', **metadata}, path)


def load_fp32(path):
    cp = torch.load(path, map_location='cpu', weights_only=False)
    m = ResNet18(cp['hidden'])
    m.load_state_dict(cp['state_dict'])
    return m.eval(), cp


def structured_prune(model, fraction):
    """Physically slice conv1 rows, BN1, and connected conv2 columns.

    The output width of each BasicBlock and all skip connections is retained.
    Every pruned block bottleneck has fewer real channels, with no masks.
    """
    widths, indices, records = [], [], []
    for i,b in enumerate(model.blocks):
        w = b.conv1.weight.detach()
        score = w.abs().mean((1,2,3))
        k = max(8, round(w.shape[0] * (1-fraction) / 8) * 8)
        ids = score.topk(k).indices.sort().values
        widths.append(k); indices.append(ids)
        records.append({'block':i, 'before':int(w.shape[0]), 'after':k,
                        'kept_indices': ids.cpu().tolist()})
    dst = ResNet18(widths)
    state = model.state_dict()
    sliced = {}
    for name,t in dst.state_dict().items():
        src = state[name]
        if name.startswith('blocks.'):
            i = int(name.split('.')[1]); ids = indices[i]
            if '.conv1.weight' in name: src = src[ids]
            elif '.bn1.' in name and src.ndim: src = src[ids]
            elif '.conv2.weight' in name: src = src[:,ids]
        sliced[name] = src.clone()
    dst.load_state_dict(sliced)
    assert not any('mask' in n for n,_ in dst.named_buffers())
    return dst, records
