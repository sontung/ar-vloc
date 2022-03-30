cd vloc_workspace_retrieval
colmap feature_extractor --database_path db.db --image_path images --image_list_path test_images.txt --ImageReader.single_camera 1 --ImageReader.camera_model 'PINHOLE' --ImageReader.camera_params '2600, 2600, 1134, 2016'
colmap matches_importer --database_path db.db --match_list_path retrieval_pairs.txt
colmap vocab_tree_matcher --database_path db.db --VocabTreeMatching.vocab_tree_path vocab_tree_flickr100K_words32K.bin --VocabTreeMatching.match_list_path test_images.txt
#colmap mapper --database_path db.db --image_path images --input_path sparse --output_path new
colmap image_registrator --database_path db.db --input_path sparse --output_path new --Mapper.ba_refine_focal_length 0 --Mapper.ba_refine_extra_params 0
colmap bundle_adjuster --input_path new --output_path new
colmap model_converter --input_path new --output_path new --output_type TXT
