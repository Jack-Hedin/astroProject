#!/usr/bin/env python
import sys, os, argparse

ap = argparse.ArgumentParser()
ap.add_argument('--print')
ap.add_argument('--num', type=int)
a = ap.parse_args()
print(a.print)
