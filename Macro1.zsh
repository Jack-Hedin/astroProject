#!/bin/zsh
for f in {200,300}; do
echo running for $f frames
for i in {1..6}; do
  zsh run_script.zsh --frames $f --seed $RANDOM --output "F${f}_$i"
done
cd MegaRun/
for dir in F*/; do
cd $dir
cl < ../../peak.cl
cd ../.
done
mv wsnr.txt WSNR_$f.txt
mkdir run_frames$f
mv F${f}_?/ run_frames$f/
cd ..
done

