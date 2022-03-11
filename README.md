# Visual localization for augmented reality

## main script = `retrieval_based.py`

## how to use colmap
1. feature extraction with `shared for all images`
2. feature matching with `sequential`, check `loop detection` `quadratic overlap` and `vocab tree path`
3. reconstruction: uncheck `multiple models`
4. dense reconstruction: in `stereo`, set `max image size` to 1000, `cache size` to 8 (do the same for `fusion`)

## problems
1. smoothness fails when the center is wrong.
2. post optim does not robustly solve this issue.
3. how to detect the wrong center?

## to do list
1. normalizing quat and trans before clustering
2. can SFM localize difficult image?
3. qap
   3. visualize with non-qap
   4. post-optim filter with a few iterations
4. retrieval
   1. check matches with h mat from "two view geometries"
   
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