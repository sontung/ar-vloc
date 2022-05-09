# Visual localization for augmented reality

## to do
- share one matrix for query image across all pipeline
- profile matching pairs

## main script = `retrieval_based.py`

## how to use colmap
1. feature extraction with `shared for all images`
2. feature matching with `sequential`, check `loop detection` `quadratic overlap` and `vocab tree path`
3. reconstruction: uncheck `multiple models`
4. dense reconstruction: in `stereo`, set `max image size` to 1000, `cache size` to 8 (do the same for `fusion`)
   
## build LT-pnp
1. `cmake .. -DPYTHON_EXECUTABLE=$(which python)`