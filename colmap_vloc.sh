colmap feature_extractor --database_path database.db --image_path test_line --image_list_path test_images.txt
colmap exhaustive_matcher --database_path database.db
colmap image_registrator --database_path database.db --input_path /home/sontung/work/ar-vloc/sfm_ws_hblab --output_path new_model/
colmap bundle_adjuster --input_path . --output_path new_model
colmap model_converter --input_path new_model --output_path new_model --output_type TXT