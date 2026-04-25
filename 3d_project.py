#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: csarasty
"""

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import open3d as o3d
from transformers import GLPNImageProcessor, GLPNForDepthEstimation


## Getting feature extractor 

feature_extractor = GLPNImageProcessor.from_pretrained("vinvino02/glpn-nyu")
model = GLPNForDepthEstimation.from_pretrained("vinvino02/glpn-nyu")

image = Image.open('./Downloads/car.jpg')
new_height =  480 if image.height > 480 else image.height
new_height += (new_height % 32)
new_width = int(new_height *  image.width /image.height)
diff = new_width % 32

new_width = new_width - diff if diff < 16 else new_width + 32 - diff
new_size = (new_width, new_height)
image = image.resize(new_size)

#Apply model 

inputs = feature_extractor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    predicted_depth = outputs.predicted_depth
    
#postprocessing

pad = 16
output = predicted_depth.squeeze().cpu().numpy()*1000
output = output[pad:-pad, pad:-pad]
image = image.crop((pad, pad, image.width - pad, image.height - pad))

# visualize 

fig, ax = plt.subplots(1,2)
ax[0].imshow(image)
ax[0].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
ax[1].imshow(output, cmap='plasma')
ax[1].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
plt.tight_layout()
plt.pause(5)


#Point Cloud Generation 

width, height = image.size

depth_image_raw  = (output * 255 / np.max(output)).astype('uint8')

# Resize depth_image_raw to match the RGB image dimensions
depth_image = Image.fromarray(depth_image_raw)
depth_image = depth_image.resize((width, height), Image.LANCZOS) # Resize to match image (width, height)
depth_image = np.array(depth_image)

image = np.array(image)

# create rgbd image
depth_o3d = o3d.geometry.Image(depth_image)
image_o3d = o3d.geometry.Image(image)
rgb_to_id = o3d.geometry.RGBDImage.create_from_color_and_depth(image_o3d,
depth_o3d, convert_rgb_to_intensity=False)

# creating a camera 
camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
camera_intrinsic.set_intrinsics(width, height, 500, 500, width/2, height/2)

#create and visualizing point cloud

pcd_raw = o3d.geometry.PointCloud.create_from_rgbd_image(rgb_to_id, camera_intrinsic)
o3d.visualization.draw_geometries([pcd_raw])

# postprocessing  3d point cloud 

#outliers removal 
cl, ind = pcd_raw.remove_statistical_outlier(nb_neighbors=20, std_ratio=6.0)
pcd = pcd_raw.select_by_index(ind)

#estimate normals
pcd.estimate_normals()
pcd.orient_normals_to_align_with_direction()
o3d.visualization.draw_geometries([pcd])


#surface reconstruction 
#use poison reconstruction 
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth = 10, n_threads=1)[0]
#rotate mesh
rotation  = mesh.get_rotation_matrix_from_xyz((np.pi, 0, 0))

#visualize mesh
o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)



# export Mesh 
#uncomment to export 
#o3d.io.write_triangle_mesh('./project3d.obj', mesh)







