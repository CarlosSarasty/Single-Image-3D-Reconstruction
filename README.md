# Overview
This project generates a 3D point cloud and surface mesh from a single RGB image using a learned depth estimation model and geometric back-projection.
The pipeline combines a pretrained monocular depth model, GLPN, with 3D processing tools from Open3D to produce a pseudo-3D reconstruction of the scene.
Unlike traditional photogrammetry, this approach does not use multiple views or triangulation. Instead, depth is inferred from learned visual priors.

# What it does
- Loads an input image
- Predicts a depth map using a pretrained model
- Converts the RGB image and depth map into an RGB-D representation
- Projects pixels into 3D using a pinhole camera model
- Generates a point cloud
- Removes outliers and estimates normals
- Reconstructs a surface mesh using Poisson reconstruction
- Visualizes and exports the mesh


# Output
- Depth map visualization
- 3D point cloud
- Reconstructed mesh (.obj)
