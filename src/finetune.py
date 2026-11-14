"""Recovery fine-tuning entry point for QAT and both pruning methods."""
import sys
from cli import parser
from qat import run_qat
from prune_unstructured import run_unstructured
from prune_structured import run_structured
if __name__=='__main__':
    if len(sys.argv)<2 or sys.argv[1] not in ('qat','unstructured','structured'):
        raise SystemExit('Usage: python src/finetune.py qat|unstructured|structured [options]')
    kind=sys.argv.pop(1)
    {'qat':run_qat,'unstructured':run_unstructured,'structured':run_structured}[kind](parser(kind).parse_args())
