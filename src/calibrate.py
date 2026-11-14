"""Explicit calibration entry point; prepares, calibrates and converts PTQ."""
from cli import parser
from ptq import run_ptq
if __name__=='__main__':run_ptq(parser('ptq').parse_args())
