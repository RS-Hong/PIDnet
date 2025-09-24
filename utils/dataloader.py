import cv2
import torch
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
from utils.transforms import RandomResize, RandomCrop, RandomFlip, PhotoMetricDistortion, GenerateEdge
import os

class SegmentationDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, random, dataset_path):
        super(SegmentationDataset, self).__init__()
        self.annotation_lines = annotation_lines
        self.length = len(annotation_lines)
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.random = random
        self.dataset_path = dataset_path

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        annotation_line = self.annotation_lines[index]
        name = annotation_line.split()[0]

        jpg_path = os.path.join(self.dataset_path, "VOC2007/JPEGImages", name + ".tif")
        png_path = os.path.join(self.dataset_path, "VOC2007/SegmentationClass", name + ".tif")
        if not os.path.exists(jpg_path) or not os.path.exists(png_path):
            raise FileNotFoundError(f"Image or label not found: {jpg_path}, {png_path}")

        # 使用 PIL 加载 TIFF 图像和标签
        jpg = Image.open(jpg_path).convert('RGB')  # 确保图像为 3 通道 uint8
        png = Image.open(png_path)  # uint16 标签

        # 确保图像尺寸为 512x512
        if jpg.size != (512, 512):
            jpg = jpg.resize((512, 512), Image.BILINEAR)
            png = png.resize((512, 512), Image.NEAREST)

        if self.random:
            pipeline = [
                RandomResize(scale=(512, 512), ratio_range=(1.0, 2.0)),
                RandomCrop(crop_size=self.input_shape),
                RandomFlip(prob=0.5),
                PhotoMetricDistortion()
            ]
            for trans in pipeline:
                jpg, png = trans(jpg, png)

        # 确保最终尺寸匹配 input_shape
        if jpg.size != tuple(self.input_shape):
            jpg = jpg.resize(self.input_shape, Image.BILINEAR)
            png = png.resize(self.input_shape, Image.NEAREST)

        jpg = np.array(jpg, dtype=np.float32)
        mean = np.array([123.675, 116.28, 103.53])
        std = np.array([58.395, 57.12, 57.375])
        jpg = (jpg - mean) / std

        png = np.array(png, dtype=np.uint16)
        # 检查标签值范围
        unique_values = np.unique(png)
        if np.any(png > self.num_classes) and not np.all(png[png > self.num_classes] == 255):
            print(f"Warning: Invalid label values found in {png_path}: {unique_values}")
        # 映射非法值到 255 并转换为 uint8
        png[png >= self.num_classes] = 255
        png = png.astype(np.uint8)

        edge = GenerateEdge(edge_width=4)(png)
        jpg = np.transpose(jpg, [2, 0, 1])
        return torch.from_numpy(jpg).float(), torch.from_numpy(png).long(), torch.from_numpy(edge).float()

def seg_dataset_collate(batch):
    images, pngs, edges = [], [], []
    for img, png, edge in batch:
        images.append(img)
        pngs.append(png)
        edges.append(edge)
    images = torch.stack(images)
    pngs = torch.stack(pngs)
    edges = torch.stack(edges)
    return images, pngs, edges