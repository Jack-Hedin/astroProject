./Broadband_solve.py --niter 50 --whiten 1 --restart 0 --output 'run1'
./Broadband_solve.py --niter 150 --whiten 2 --restart 1 --output 'run2' --estimate_path 'C:/Users/jackr/OneDrive/Documents/Desktop/Hedin/run1/estimate.fits'
./Broadband_solve.py --niter 0 --whiten 2 --restart 1 --output 'run3' --estimate_path 'C:/Users/jackr/OneDrive/Documents/Desktop/Hedin/run2/estimate.fits' --filter 1 --support 1

./detect_sources_new.py 'C:/users/jackr/OneDrive/Documents/Desktop/Hedin/run3/' --center 255.5 251

./Broadband_solve.py --niter 0 --whiten 2 --restart 1 --output 'run5' --estimate_path 'C:/Users/jackr/OneDrive/Documents/Desktop/Hedin/run4/estimate.fits' --companion 1 --support 1



