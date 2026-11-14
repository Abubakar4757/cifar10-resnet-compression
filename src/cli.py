"""Shared arguments; separate entry-point scripts remain independently runnable."""
import argparse


def parser(kind):
    p=argparse.ArgumentParser(description=f'Assignment 3 {kind}')
    p.add_argument('--data',default='data')
    p.add_argument('--output',default='artifacts')
    if kind!='train':p.add_argument('--baseline',default='artifacts/baseline_fp32.pt')
    if kind=='train':p.add_argument('--epochs',type=int,default=50)
    elif kind=='ptq':p.add_argument('--calibration',type=int,default=1024)
    elif kind=='qat':p.add_argument('--epochs',type=int,default=6)
    elif kind=='unstructured':p.add_argument('--epochs',type=int,default=3)
    elif kind=='structured':
        p.add_argument('--epochs',type=int,default=8)
        p.add_argument('--fraction',type=float,default=.375)
    return p
