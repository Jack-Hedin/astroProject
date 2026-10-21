#!/bin/zsh
python3 Broadband_solve.py --niter 50 --whiten 1 --restart 0 --output 'run1'
python3 Broadband_solve.py --niter 150 --whiten 2 --restart 1 --output 'run2' --estimate_path './run1/estimate.fits'
python3 Broadband_solve.py --niter 0 --whiten 2 --restart 1 --output 'run3' --estimate_path './run2/estimate.fits' --filter 1 --support 1

python 3detect_sources_new.py './run3/' --center 255.5 251

#./Broadband_solve.py --niter 0 --whiten 2 --restart 1 --output 'run4' --estimate_path './run3/estimate.fits' --companion 1 --support 1



