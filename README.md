# Metashape 360 to LichtFeld Studio Converter

Converts Metashape 360° equirectangular camera poses and sparse point clouds to the NeRF/Blender transforms.json format supported by LichtFeld Studio. Also supports pinhole and fisheye cameras.

**Video tutorial:** https://youtu.be/b1Olu_IU1sM

## Scripts

**metashape_360_lfs.py** - Standalone script for use outside Metashape. Takes exported XML and PLY files as input.

**metashapepro_360_lfs.py** - Runs directly inside Metashape Pro. Adds a menu item under Scripts > Export to LichtFeld Format.

## Standalone Usage

```bash
pip install numpy open3d

python metashape_360_lfs.py --images ./images/ --xml cameras.xml --ply sparse.ply --output ./output/
```

Export from Metashape first:
- File > Export > Export Cameras... (save as XML)
- File > Export > Export Point Cloud... (save as PLY)

### Options

| Flag | Description |
|------|-------------|
| `--images` | Directory containing source images |
| `--xml` | Metashape cameras.xml file |
| `--ply` | Sparse point cloud PLY (optional) |
| `--output` | Output directory (default: same as XML) |
| `--no-fix-rotation` | Disable the 90° orientation fix |

## Metashape Pro Usage

Copy `metashapepro_360_lfs.py` to your Metashape scripts folder and restart, then use Scripts > Export to LichtFeld Format.

Or run it directly via Tools > Run Script.

## Coordinate Transform

Handles the coordinate system conversion from Metashape to LichtFeld, including component transforms from rotated/translated chunks.

## Output

```
output/
├── transforms.json    # Camera poses and intrinsics
└── pointcloud.ply     # Transformed sparse point cloud
```

The transforms.json follows the nerfstudio format with per-frame intrinsics and a pointcloud path.
