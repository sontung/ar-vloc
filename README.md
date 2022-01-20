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
## build LT-pnp

1. `cmake .. -DPYTHON_EXECUTABLE=$(which python)`
