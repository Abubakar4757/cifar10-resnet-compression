"""Physical channel reconstruction and recovery fine-tuning."""
from pathlib import Path
from cli import parser
from models import load_fp32, save_fp32, structured_prune
from data import Data, accuracy
from engine import initialize, train, write_json, sparsity


def run_structured(args):
    device=initialize();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    m,_=load_fp32(args.baseline)
    before=sparsity(m)
    m,record=structured_prune(m,args.fraction)
    data=Data(args.data,device)
    acc=accuracy(m.to(device),data)
    m,summary=train(m,data,out,'structured',args.epochs,.02)
    save_fp32(m,out/'structured_fp32.pt',pruning_record=record,**summary)
    summary.update({'fraction':args.fraction,'before_finetune_val':acc,
                    'original':before,'pruned':sparsity(m),'channels':record})
    write_json(out/'structured_summary.json',summary)
    return m


if __name__=='__main__':run_structured(parser('structured').parse_args())
