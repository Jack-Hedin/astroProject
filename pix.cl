if (access("ls150")) cd ls150
boxcar estimate[*,*,1] z1 3 3
boxcar estimate[*,*,2] z2 3 3
boxcar estimate[*,*,3] z3 3 3
boxcar estimate[*,*,4] z4 3 3
boxcar estimate[*,*,5] z5 3 3
boxcar estimate[*,*,6] z6 3 3
boxcar estimate[*,*,7] z7 3 3
boxcar estimate[*,*,8] z8 3 3
boxcar estimate[*,*,9] z9 3 3
boxcar estimate[*,*,10] z10 3 3
imstack z* zz
imarith zz * 9 estimate_box
imdel z1,z2,z3,z4,z5,z6,z7,z8,z9,z10,zz
listpix estimate_box[270:270,257:257,*] > delta_estimate.txt
listpix object[270:270,257:257,*] > delta_true.txt
logout
