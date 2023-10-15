
import numpy as np
from torchvision.transforms import ToTensor
from PIL import Image
from zoedepth.utils.misc import get_image_from_url, colorize
import torch

from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config
from pprint import pprint
import open3d as o3d


def get_intrinsics(H,W):
    """
    Intrinsics for a pinhole camera model.
    Assume fov of 55 degrees and central principal point.
    """
    f = 0.5 * W / np.tan(0.5 * 55 * np.pi / 180.0)
    cx = 0.5 * W
    cy = 0.5 * H
    return np.array([[f, 0, cx],
                     [0, f, cy],
                     [0, 0, 1]])

def depth_to_points(depth, R=None, t=None):

    K = get_intrinsics(depth.shape[1], depth.shape[2])
    Kinv = np.linalg.inv(K)
    if R is None:
        R = np.eye(3)
    if t is None:
        t = np.zeros(3)

    # M converts from your coordinate to PyTorch3D's coordinate system
    M = np.eye(3)
    M[0, 0] = -1.0
    M[1, 1] = -1.0

    height, width = depth.shape[1:3]

    x = np.arange(width)
    y = np.arange(height)
    coord = np.stack(np.meshgrid(x, y), -1)
    coord = np.concatenate((coord, np.ones_like(coord)[:, :, [0]]), -1)  # z=1
    coord = coord.astype(np.float32)
    # coord = torch.as_tensor(coord, dtype=torch.float32, device=device)
    coord = coord[None]  # bs, h, w, 3

    D = depth[:, :, :, None, None]
    # print(D.shape, Kinv[None, None, None, ...].shape, coord[:, :, :, :, None].shape )
    pts3D_1 = D * Kinv[None, None, None, ...] @ coord[:, :, :, :, None]
    # pts3D_1 live in your coordinate system. Convert them to Py3D's
    pts3D_1 = M[None, None, None, ...] @ pts3D_1
    # from reference to targe tviewpoint
    pts3D_2 = R[None, None, None, ...] @ pts3D_1 + t[None, None, None, :, None]
    # pts3D_2 = pts3D_1
    # depth_2 = pts3D_2[:, :, :, 2, :]  # b,1,h,w
    return pts3D_2[:, :, :, :3, 0][0]


torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=True) 

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cpu":
    print("WARNING: Running on CPU. This will be slow. Check your CUDA installation.")

print("*" * 20 + " Testing zoedepth " + "*" * 20)
conf = get_config("zoedepth", "infer")


print("Config:")
pprint(conf)

model = build_model(conf).to(DEVICE)
model.eval()

print("-"*20 + " Testing on an indoor scene from url " + "-"*20)

# Test img
# img_path = '/home/yuxuan/project/ZoeDepth/data/imageFiles/flat_packBags_00/image_00000.jpg'
# img_path = '/home/yuxuan/project/ZoeDepth/data/behave/Date01_Sub01_tablesmall_move/t0003.000/k2.color.jpg'
img_path = '/home/yuxuan/project/ZoeDepth/data/shhq/images/11801.jpg'
img = Image.open(img_path)
orig_size = img.size
X = ToTensor()(img)
X = X.unsqueeze(0).to(DEVICE)

print("X.shape", X.shape)
print("predicting")

with torch.no_grad():
    out = model.infer(X).cpu()

# or just, 
# out = model.infer_pil(img)

pts3d = depth_to_points(out[0].numpy())
pts3d = pts3d.reshape(-1, 3)
image = np.array(img)
colors = image.reshape(-1, 3) / 255

# create o3d point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pts3d)
pcd.colors = o3d.utility.Vector3dVector(colors)

# write o3d point cloud
o3d.io.write_point_cloud('sample/'+img_path.split('/')[7]+"_00000.ply", pcd)
# o3d.visualization.draw_geometries([pcd])

print("output.shape", out.shape)
pred = Image.fromarray(colorize(out))
# Stack img and pred side by side for comparison and save
pred = pred.resize(orig_size, Image.ANTIALIAS)
stacked = Image.new("RGB", (orig_size[0]*2, orig_size[1]))
stacked.paste(img, (0, 0))
stacked.paste(pred, (orig_size[0], 0))

stacked.save('sample/'+img_path.split('/')[7]+"_00000.png")
print("saved pred.png")


model.infer_pil(img, output_type="pil").save("pred_raw.png")