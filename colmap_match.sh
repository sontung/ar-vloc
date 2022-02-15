cd vloc_workspace
colmap feature_extractor --database_path database.db --image_path images --image_list_path test_images.txt
colmap exhaustive_matcher --database_path database.db
colmap mapper --database_path database.db --image_path images --input_path sparse/0 --output_path new
colmap bundle_adjuster --input_path new --output_path new
colmap model_converter --input_path new --output_path new --output_type TXT
