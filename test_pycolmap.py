import pycolmap
reconstruction = pycolmap.Reconstruction("vloc_workspace_retrieval/new")
print(reconstruction.summary())

for image_id, image in reconstruction.images.items():
    print(image_id, image.name)

# for point3D_id, point3D in reconstruction.points3D.items():
#     print(point3D_id, point3D)
#
# for camera_id, camera in reconstruction.cameras.items():
#     print(camera_id, camera)