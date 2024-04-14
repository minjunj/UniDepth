import torch
import numpy as np
from PIL import Image
import sys

from unidepth.utils import colorize, image_grid

def demo(model, image_path, intrinsics_path, points):
    # Load your image
    rgb = np.array(Image.open(image_path))
    rgb_torch = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0  # Normalize and convert to tensor

    # Load intrinsics
    intrinsics_torch = torch.from_numpy(np.load(intrinsics_path))

    # Predict depth
    predictions = model.infer(rgb_torch.unsqueeze(0), intrinsics_torch.unsqueeze(0))

    # Process prediction
    depth_pred = predictions["depth"].squeeze().cpu().numpy()

    # Colorize depth prediction
    depth_pred_col = colorize(depth_pred, vmin=0.01, vmax=10.0, cmap="magma_r")

    # Save image with prediction
    Image.fromarray((depth_pred_col * 255).astype(np.uint8)).save("output_depth.png")

    # Output depths at specific points If don't wonder the specific points, remove this part.
    for point in points:
        x, y = point
        if 0 <= x < depth_pred.shape[1] and 0 <= y < depth_pred.shape[0]:
            depth_value = depth_pred[y, x]
            print(f"Depth at point ({x}, {y}): {depth_value:.2f} meters")
        else:
            print(f"Point ({x}, {y}) is out of the image bounds.")

    # Output diagnostics
    print("Prediction complete. Depth image saved as 'output_depth.png'.")

if __name__ == "__main__":
    print("Torch version:", torch.__version__)
    model = torch.hub.load(
        "lpiccinelli-eth/unidepth",
        "UniDepth",
        backbone="ViTL14",
        version="v1",
        pretrained=True,
        trust_repo=True,
        force_reload=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Specify the path to your image and intrinsics file
    image_path = "assets/demo/5m_2.jpg"
    intrinsics_path = "assets/demo/intrinsics.npy"

    # Specify points for depth value extraction (x, y tuples)
    points = [(750, 550), (800, 600), (870, 640)]

    demo(model, image_path, intrinsics_path, points)