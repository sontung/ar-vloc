# Visual localization for augmented reality

## to do list
1. normalizing quat and trans before clustering
3. can SFM localize difficult image?
5. qap
   3. visualize with non-qap
   4. perform ransac pnp while solving neighborhood
   5. check unary cost with pw cost (when checking bad neighborhood)

## experimental results

|             | no pw     | pw using cosine | pw using `np.var([v1, v2])` | pw using 1-cosine |
|-------------|-----------|-----------------|-----------------------------|-------------------|
| distance    | 164.66    | 165.97          | 146.5                       | 175.66            |
| accuracy    | 0.078     | 0.078           | 0.078                       | 0.078             |
| geom. cost  | 68440.125 | 67327.11        | 58809.8                     | 69330.03          |
| inlier cost | 19        | 37              | 35                          | 22                |

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