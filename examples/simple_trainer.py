import math
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import tyro
from gsplat.project_gaussians import project_gaussians
from gsplat.rasterize import rasterize_gaussians
from PIL import Image
from torch import Tensor, optim
import gsplat._torch_impl
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image


class SimpleTrainer:
    """Trains random gaussians to fit an image."""

    def __init__(
        self,
        gt_image: Tensor,
        num_points: int = 2000,
    ):
        self.device = torch.device("cuda:0")
        self.gt_image = gt_image.to(device=self.device)
        self.num_points = num_points

        fov_x = math.pi / 2.0
        self.H, self.W = gt_image.shape[0], gt_image.shape[1]
        self.focal = 0.5 * float(self.W) / math.tan(0.5 * fov_x)
        self.img_size = torch.tensor([self.W, self.H, 1], device=self.device)

        self._init_gaussians()

    def _init_gaussians(self):
        """Random gaussians"""
        bd = 2

        self.means = bd * (torch.rand(self.num_points, 3, device=self.device) - 0.5)
        self.scales = torch.rand(self.num_points, 3, device=self.device)
        d = 3
        self.rgbs = torch.rand(self.num_points, d, device=self.device)

        u = torch.rand(self.num_points, 1, device=self.device)
        v = torch.rand(self.num_points, 1, device=self.device)
        w = torch.rand(self.num_points, 1, device=self.device)

        self.quats = torch.cat(
            [
                torch.sqrt(1.0 - u) * torch.sin(2.0 * math.pi * v),
                torch.sqrt(1.0 - u) * torch.cos(2.0 * math.pi * v),
                torch.sqrt(u) * torch.sin(2.0 * math.pi * w),
                torch.sqrt(u) * torch.cos(2.0 * math.pi * w),
            ],
            -1,
        )
        self.opacities = torch.ones((self.num_points, 1), device=self.device)

        self.viewmat = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 8.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            device=self.device,
        )
        self.background = torch.zeros(d, device=self.device)

        self.means.requires_grad = True
        self.scales.requires_grad = True
        self.quats.requires_grad = True
        self.rgbs.requires_grad = True
        self.opacities.requires_grad = True
        self.viewmat.requires_grad = False

    def train(
        self,
        iterations: int = 1000,
        lr: float = 0.01,
        save_imgs: bool = False,
        B_SIZE: int = 14,
    ):
        optimizer = optim.Adam(
            [self.rgbs, self.means, self.scales, self.opacities, self.quats], lr
        )
        mse_loss = torch.nn.MSELoss()
        frames = []
        times = [0] * 3  # project, rasterize, backward
        B_SIZE = 16
        for iter in range(iterations):
            start = time.time()
            (
                xys,
                depths,
                radii,
                conics,
                compensation,
                num_tiles_hit,
                cov3d,
            ) = project_gaussians(
                self.means,
                self.scales,
                1,
                self.quats / self.quats.norm(dim=-1, keepdim=True),
                self.viewmat,
                self.focal,
                self.focal,
                self.W / 2,
                self.H / 2,
                self.H,
                self.W,
                B_SIZE,
            )
            torch.cuda.synchronize()
            times[0] += time.time() - start
            start = time.time()
            out_img = rasterize_gaussians(
                xys,
                depths,
                radii,
                conics,
                num_tiles_hit,
                torch.sigmoid(self.rgbs),
                torch.sigmoid(self.opacities),
                self.H,
                self.W,
                B_SIZE,
                self.background,
            )[..., :3]
            torch.cuda.synchronize()
            times[1] += time.time() - start
            loss = mse_loss(out_img, self.gt_image)
            optimizer.zero_grad()
            start = time.time()
            loss.backward()
            torch.cuda.synchronize()
            times[2] += time.time() - start
            optimizer.step()
            print(f"Iteration {iter + 1}/{iterations}, Loss: {loss.item()}")

            if save_imgs and iter % 5 == 0:
                frames.append((out_img.detach().cpu().numpy() * 255).astype(np.uint8))
        if save_imgs:
            # save them as a gif with PIL
            frames = [Image.fromarray(frame) for frame in frames]
            out_dir = os.path.join(os.getcwd(), "renders")
            os.makedirs(out_dir, exist_ok=True)
            frames[0].save(
                f"{out_dir}/trainingSlang.gif", # change this to prevent overwriting existing training gifs
                save_all=True,
                append_images=frames[1:],
                optimize=False,
                duration=5,
                loop=0,
            )
            # save only the last frame as a PNG 
            last_frame = frames[-1]
            out_dir = os.path.join(os.getcwd(), "renders")
            os.makedirs(out_dir, exist_ok=True)
            last_frame.save(
                f"{out_dir}/last_frameSlang.png",  # Change this to prevent overwriting existing files
                format="PNG"
            )
        totalResults = f"Total(s):\nProject: {times[0]:.3f}, Rasterize: {times[1]:.3f}, Backward: {times[2]:.3f}"
        print(totalResults)
        print(
            f"Per step(s):\nProject: {times[0]/iterations:.5f}, Rasterize: {times[1]/iterations:.5f}, Backward: {times[2]/iterations:.5f}"
        )
        file_name = "Slang_results.txt"
        with open(file_name, "a") as file:
            file.write(totalResults + "\n")


def image_path_to_tensor(image_path: Path):
    import torchvision.transforms as transforms

    img = Image.open(image_path)
    transform = transforms.ToTensor()
    img_tensor = transform(img).permute(1, 2, 0)[..., :3]
    return img_tensor

def load_image_as_tensor(image_path):
    """
    Converts image to tensor
    """
    image = Image.open(image_path)
    transform = transforms.ToTensor()
    tensor = transform(image)
    return tensor

def calculatePSNR(input, output):
    """
    Calculate PSNR between GT and final render.
    """
    mse = F.mse_loss(input, output)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0 if input.max() <= 1 else 255.0
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr.item()

def main(
    height: int = 822, #822
    width: int = 1237, #1237
    num_points: int = 2000, # orig: 100000
    save_imgs: bool = True, # change this to get the resulting image
    img_path: Optional[Path] = 'bikeImage.png',
    iterations: int = 1000,
    lr: float = 0.01,
) -> None:
    if img_path:
        gt_image = image_path_to_tensor(img_path)
    else:
        gt_image = torch.ones((height, width, 3)) * 1.0
        # make top left and bottom right red, blue
        gt_image[: height // 2, : width // 2, :] = torch.tensor([1.0, 0.0, 0.0])
        gt_image[height // 2 :, width // 2 :, :] = torch.tensor([0.0, 0.0, 1.0])

    trainer = SimpleTrainer(gt_image=gt_image, num_points=num_points)
    for i in range(99):
        trainer.train(
            iterations=iterations,
            lr=lr,
            save_imgs=save_imgs,
        )
    output_image = load_image_as_tensor('renders/last_frameSlang.png')
    output_image = output_image.permute(1, 2, 0)  # Change from (3, 822, 1237) to (822, 1237, 3)
    psnr = calculatePSNR(gt_image,output_image)
    print("PSNR: " + str(psnr))

if __name__ == "__main__":
    tyro.cli(main)
