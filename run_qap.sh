cd /home/sontung/tools/libmpopt/qapopt/build/bin
qap_dd_greedy_gen --max-batches 10 --batch-size 1 --generate 1 /home/sontung/work/ar-vloc/qap/input.dd /home/sontung/work/ar-vloc/qap/proposal.txt
qap_dd_fusion --solver qpbo-i --output /home/sontung/work/ar-vloc/qap/fused.txt /home/sontung/work/ar-vloc/qap/input.dd /home/sontung/work/ar-vloc/qap/proposal.txt
cd /home/sontung/work/ar-vloc