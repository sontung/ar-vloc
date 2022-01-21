# Visual localization for augmented reality

## to do list
1. normalizing quat and trans before clustering
2. enforcing smoothness to recover more matches
   1. if one match is found,
   2. split covisible neighbors of the point into a multi-grid
   3. do the same for the neighbors of the feature
   4. check if we can find any extra matches
3. can SFM localize difficult image?
4. combinatorics
   1. compute assignment costs
   2. cluster into k groups
   3. edge cost within this group
   4. normalize all costs
5. qap
   1. inlier thresholded
   2. consistency
   3. visualize with non-qap

## build LT-pnp
1. `cmake .. -DPYTHON_EXECUTABLE=$(which python)`

## compile QAP-DD
0. `mkdir build && conda activate env`
1. `meson setup build --python.platlibdir='/home/sontung/anaconda3/envs/env/lib/python3.9/site-packages/' --python.purelibdir='/home/sontung/anaconda3/envs/env/lib/python3.9/site-packages/'`
2. `ninja -C build`
3. `ninja -C build install`

## run QAP-DD

1. `qap_dd_greedy_gen --verbose --max-batches 1000 --batch-size 1 --generate 1 /home/sontung/work/ar-vloc/qap/input.dd /home/sontung/work/ar-vloc/qap/proposal.txt`
2. `qap_dd_fusion --solver qpbo-i --output /home/sontung/work/ar-vloc/qap/fused.txt /home/sontung/work/ar-vloc/qap/input.dd /home/sontung/work/ar-vloc/qap/proposal.txt`