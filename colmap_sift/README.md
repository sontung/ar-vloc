### How to extract SIFT using COLMAP

```
colmap feature_extractor --image_path "../sfm_ws_hblab/images" --image_list_path "train_images.txt" --database_path "train.db"
colmap feature_extractor --image_path "../Test line small" --image_list_path "test_images.txt" --database_path "test_small.db" --ImageReader.default_focal_length_factor=0.64484
```

### How to matching
```
colmap feature_extractor --database_path database.db --image_path test_line --image_list_path test_images.txt
colmap exhaustive_matcher --database_path database.db
```

### COLMAP compile
instruction: https://colmap.github.io/install.html

issue: https://github.com/colmap/colmap/issues/188