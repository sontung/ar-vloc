# Visual localization for augmented reality

## to do
- try `how` descriptors

## main script = `retrieval_based*.py`

## after changing retrieval pairs
1. have to perform matches again
2. thus, delete `matches.h5` `matches_1.pkl` `id2matches.pkl`
3. delete `pairs.txt`

## how to evaluate
1. modify global variables in `evaluation_utils.py`.
2. copy `images.txt` and `points3d.txt` to workspace directory (export to txt files using colmap if only bin files are available).
3. run `evaluation_utils.py`.
4. download 7 scenes images from Microsoft, and unzip everything inside the downloaded file.
5. run `evaluate_7scenes.py`.
6. change camera params if necessary.

## evaluate using 7scenes dataset from [this repo](https://github.com/tsattler/visloc_pseudo_gt_limitations)
1. edit the config
2. run `python evaluate_estimates.py config_7scenes_sfm_pgt.json`

## how to use colmap
1. feature extraction with `shared for all images`
2. feature matching with `sequential`, check `loop detection` `quadratic overlap` and `vocab tree path`
3. reconstruction: uncheck `multiple models`
4. dense reconstruction: in `stereo`, set `max image size` to 1000, `cache size` to 8 (do the same for `fusion`)
   
## build LT-pnp
1. `cmake .. -DPYTHON_EXECUTABLE=$(which python)`