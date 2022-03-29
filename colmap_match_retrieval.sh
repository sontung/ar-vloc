cd vloc_workspace_retrieval
colmap feature_extractor --database_path db.db --image_path images --image_list_path test_images.txt
colmap matches_importer --database_path db.db --match_list_path retrieval_pairs.txt
colmap mapper --database_path db.db --image_path images --input_path sparse --output_path new
colmap image_registrator --database_path db.db --input_path sparse --output_path new
colmap bundle_adjuster --input_path new --output_path new
colmap model_converter --input_path new --output_path new --output_type TXT
