# Visual localization for augmented reality

## to do
- share one matrix for query image across all pipeline
- convert all to id only (currently mixed name and id)
- debug mode for cross compare
- what to do when cross compare does not find any matches? second time matching.
- d2 mask in cc

## main script = `retrieval_based.py`

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