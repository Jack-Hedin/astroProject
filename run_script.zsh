#!/bin/zsh
python3 Broadband_solve.py --niter 50 --whiten 1 --restart 0 --output 'ls50'
python3 Broadband_solve.py --niter 150 --whiten 2 --restart 1 --output 'ls150' --estimate_path './ls50/estimate.fits'
python3 Broadband_solve.py --niter 0 --whiten 2 --restart 1 --output 'ls150_MF' --estimate_path './ls150/estimate.fits' --filter 1 --support 1

python3 detect_sources_new.py './ls150_MF/' --center 255.5 251
cl < pix.cl

python3 Broadband_solve.py --niter 0 --whiten 2 --restart 1 --output 'ls_c' --estimate_path './ls150/estimate.fits' --companion 1 --support 1

cp ls_c/companion_spectrum* . 
cp ls150/delta_* .

python3 plot_companion_spectrum.py