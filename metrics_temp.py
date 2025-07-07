#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import matplotlib.pyplot as plt
from matplotlib import colors
from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim,l1_loss
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
import cv2
import numpy as np
def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names

# def l1_loss(network_output, gt):
#     return torch.abs((network_output - gt)).mean()

def create_mask(image_path):
    """
    创建一个单通道（灰度）图像作为mask，其中背景色（像素值为71）为黑色（0），其余有效区域为白色（255）。
    """
    depth = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)[:,:,0]  # 读取灰度图像

    # mask = np.zeros_like(depth, dtype=np.uint8)  # 创建一个与原图尺寸相同、类型为uint8的全0数组作为mask
    mask = (depth != 65504.0).astype(np.bool_)  # 创建布尔类型的mask
    # return mask  # 将非背景色（非71）的位置设为白色（255）
    # mask = np.dstack((mask, mask, mask))
    return mask

def save_temperature_images(image, gt_image, image_name,output_dir, min_temp=10, max_temp=50 ):
    # depth_folder = '/home/ps/code/gaussian-mesh-splatting/data/Buildings_5_times/depths'
    # depth_path = os.path.join(depth_folder, image_name.split('.')[0] + '0001.exr')
    # mask = create_mask(depth_path)

    # 假设温度数据在第一个通道
    gt_image_temp = (gt_image[:, 1, :, :] * (max_temp - min_temp) + min_temp)
    image_temp = (image[:, 1, :, :] * (max_temp - min_temp) + min_temp)

    image_np = image_temp.squeeze().cpu().numpy()
    gt_image_np = gt_image_temp.squeeze().cpu().numpy()

    # print('valid pixel:', np.sum(mask))

    # 应用遮罩到图像和gt图像
    masked_image = image_np
    masked_gt_image = gt_image_np
    # print('image valid pixel:', np.sum(masked_image.mask))
    # print('gt_image valid pixel:', np.sum(masked_gt_image.mask))
    # 计算绝对温度差值
    abs_diff_image = np.abs(image_np - gt_image_np)
    # print('abs_diff_image valid pixel:', np.sum(abs_diff_image.mask))
    # 创建差异图的最小和最大值
    min_diff = np.min(abs_diff_image)
    max_diff = np.max(abs_diff_image)

    # 计算MAE，只考虑遮罩区域内的像素
    # mae_temp = np.mean(abs_diff_image.data)
    # print(image_name, 'MAE_temp:', mae_temp)
    # valid_abs_diff = abs_diff_image.compressed()
    # mae_temp = np.mean(valid_abs_diff)
    # print(image_name, 'MAE_temp:', mae_temp)
    valid_abs_diff = abs_diff_image
    mae_temp = np.mean(valid_abs_diff)
    # print(image_name, 'MAE_temp:', mae_temp)
    # 可视化
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # GT temperature visualization
    # min_gt = np.min(masked_gt_image)
    # max_gt = np.max(masked_gt_image) 
    im1 = axs[0].imshow(masked_gt_image, cmap='gray', vmin=min_temp, vmax=max_temp)
    axs[0].set_title('GT Temperature')
    cbar1 = fig.colorbar(im1, ax=axs[0],shrink=0.6, orientation='vertical', label='Temperature (°C)')
    
    # Difference visualization
    im2 = axs[1].imshow(abs_diff_image, cmap='coolwarm', vmin=min_diff, vmax=max_diff)
    axs[1].set_title('Absolute Temperature Difference')
    cbar2 = fig.colorbar(im2, ax=axs[1],shrink=0.6, orientation='vertical', label='Absolute Temperature Difference (°C)')
    axs[1].text(0.5, -0.2, f'MAE: {mae_temp:.4f}'+'(°C)', transform=axs[1].transAxes, ha='center')
    
    # Rendered temperature visualization
    # min_rendered = np.min(masked_image)
    # max_rendered = np.max(masked_image)
    im3 = axs[2].imshow(masked_image, cmap='gray', vmin=min_temp, vmax=max_temp)
    axs[2].set_title('Rendered Temperature')
    cbar3 = fig.colorbar(im3, ax=axs[2],shrink=0.6, orientation='vertical', label='Temperature (°C)')

    plt.tight_layout()

    # 保存图像
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    fig.savefig(os.path.join(output_dir, f'{image_name}_temperature_comparison.png'))
    # im2.savefig(os.path.join(output_dir, f'{image_name}_temperature_difference.png'))
    plt.close(fig)

    return mae_temp

def temp_mae(image,gt_image,image_name,min_temp = -15.0,max_temp = 25.0):

    depth_folder = '/home/ps/code/gaussian-mesh-splatting/data/Buildings_5_times/depths'
    depth_path = os.path.join(depth_folder, image_name.split('.')[0] + '0001.exr')
    mask = create_mask(depth_path)

    gt_image_temp = (gt_image*(max_temp - min_temp)+min_temp)
    image_temp = (image*(max_temp - min_temp)+min_temp)

    image_np = image_temp.squeeze().cpu().numpy()  # 移除单维度并转换为numpy数组
    gt_image_np = gt_image_temp.squeeze().cpu().numpy()

    print('valid pixel:',np.sum(mask))
    
    # 扩展mask以匹配图像的通道数
    mask_expanded =np.vstack((mask[None, :, :],mask[None, :, :],mask[None, :, :]))  # 在第三个维度（通道维度）上扩展mask

    # 应用遮罩到图像和gt图像
    image_masked = image_np[mask_expanded]
    gt_image_masked = gt_image_np[mask_expanded]

    # 将图像和gt图像转换为torch张量
    image_tensor = torch.from_numpy(image_masked).unsqueeze(0)
    gt_image_tensor = torch.from_numpy(gt_image_masked).unsqueeze(0)

    # 计算MAE，只考虑遮罩区域内的像素
    print('valid temp:',(image_tensor[0] - gt_image_tensor[0]).shape[0]/3)
    print('render temp:',max(image_tensor[0]))
    print('gt temp:',max(gt_image_tensor[0]))
    mae_temp = torch.mean(torch.abs(image_tensor - gt_image_tensor)).item()
    print(image_name,'MAE_temp:',mae_temp)
    
    return mae_temp
    

def evaluate(gs_type, model_paths):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    for scene_dir in model_paths:
        #try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            test_dir = Path(scene_dir) / "test"

            for method in os.listdir(test_dir):
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}

                method_dir = test_dir / method
                gt_dir = method_dir/ "gt"
                vis_dir = method_dir/ "temp_viz"
                renders_dir = method_dir / f"renders{gs_type}"
                renders, gts, image_names = readImages(renders_dir, gt_dir)

                ssims = []
                psnrs = []
                lpipss = []
                mae_temp = []
                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    mae_temp.append(save_temperature_images(renders[idx], gts[idx],image_names[idx],vis_dir))
                    ssims.append(ssim(renders[idx], gts[idx]))
                    psnrs.append(psnr(renders[idx], gts[idx]))
                    # lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))
                print("  MAE_Temp : {:>12.7f}".format(torch.tensor(mae_temp).mean(), ".5"))
                print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
                # print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
                
                print("")

                full_dict[scene_dir][method].update({"MAE_Temp": torch.tensor(mae_temp).mean().item(),
                                                        "SSIM": torch.tensor(ssims).mean().item(),
                                                        "PSNR": torch.tensor(psnrs).mean().item()})
                                                        # "LPIPS": torch.tensor(lpipss).mean().item()})
                per_view_dict[scene_dir][method].update({"MAE": {name: mae for mae, name in zip(torch.tensor(mae_temp).tolist(), image_names)},
                                                            "SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                            "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)}})
                                                            # "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)}})

            with open(scene_dir + f"/results{gs_type}.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + f"/per_view{gs_type}.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
        #except:
        #    print("Unable to compute metrics for model", scene_dir)

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Metrics script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default='/home/yk98/Themal_TempGS-master/datasets/output_feicuiwan0707/s1_all_real_smooth_0926_copy')
    parser.add_argument('--gs_type', type=str, default="")
    args = parser.parse_args()
    evaluate(args.gs_type, args.model_paths)
