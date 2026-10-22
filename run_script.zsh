#!/bin/zsh
python3 Broadband_solve.py --niter 50 --whiten 1 --restart 0 --output 'run1'
python3 Broadband_solve.py --niter 200 --whiten 2 --restart 1 --output 'run2' --estimate_path './run1/estimate.fits'
python3 Broadband_solve.py --niter 0 --whiten 2 --restart 1 --output 'run3' --estimate_path './run2/estimate.fits' --filter 1 --support 1

python3 detect_sources_new.py './run2/' --center 255.5 251
cl < pix.cl

python3 Broadband_solve.py --niter 0 --whiten 2 --restart 1 --output 'run4' --estimate_path './run2/estimate.fits' --companion 1 --support 1

cp run4/companion_spectrum* . 
cp run2/delta_* .

python3 plot_companion_spectrum.py