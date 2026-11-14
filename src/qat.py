"""Quantization-aware fine-tuning from the pretrained FP32 baseline."""
from pathlib import Path
import torch
from cli import parser
from models import load_fp32
from data import Data
from engine import initialize, quant_config, save_int8, train, write_json


def run_qat(args):
    device=initialize();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    m,_=load_fp32(args.baseline);m.train();m.fuse(qat=True)
    m.qconfig=quant_config(qat=True)
    torch.ao.quantization.prepare_qat(m,inplace=True)
    data=Data(args.data,device)
    m,summary=train(m,data,out,'qat',args.epochs,.005,qat=True)
    torch.save({'state_dict':m.state_dict(),'hidden':m.hidden,'summary':summary},out/'qat_prepared.pth')
    q=torch.ao.quantization.convert(m.cpu().eval(),inplace=False)
    save_int8(q,out/'qat_int8.pt')
    from export_onnx import export_quantized
    export_quantized(q,out/'qat_int8.onnx')
    summary.update({'backend':torch.backends.quantized.engine,
                    'bn_frozen_from_epoch':3,'observers_frozen_from_epoch':4,
                    'fake_quant_modules':sum(isinstance(v,torch.ao.quantization.FakeQuantize) for v in m.modules())})
    write_json(out/'qat_summary.json',summary)
    return q


if __name__=='__main__':run_qat(parser('qat').parse_args())
