"""Train and save the original FP32 CIFAR-10 ResNet-18."""
from cli import parser
from engine import run_baseline
if __name__=='__main__':run_baseline(parser('train').parse_args())
