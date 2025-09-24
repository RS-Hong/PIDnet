import cv2
import numpy as np
from PIL import Image, ImageOps
import random

class RandomResize(object):
    def __init__(self, scale=(512, 512), ratio_range=(1.0, 2.0)):
        self.scale = scale
        self.ratio_range = ratio_range

    def __call__(self, img, seg):
        ratio = random.uniform(*self.ratio_range)
        w, h = int(self.scale[0] * ratio), int(self.scale[1] * ratio)
        img = img.resize((w, h), Image.BILINEAR)
        seg = seg.resize((w, h), Image.NEAREST)
        return img, seg

class RandomCrop(object):
    def __init__(self, crop_size=(512, 512), cat_max_ratio=0.75):
        self.crop_size = crop_size
        self.cat_max_ratio = cat_max_ratio

    def __call__(self, img, seg):
        w, h = img.size
        if w < self.crop_size[0] or h < self.crop_size[1]:
            pad_w = max(0, self.crop_size[0] - w)
            pad_h = max(0, self.crop_size[1] - h)
            img = ImageOps.expand(img, (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2), fill=0)
            seg = ImageOps.expand(seg, (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2), fill=255)
            w, h = img.size

        if w == self.crop_size[0] and h == self.crop_size[1]:
            return img, seg

        x1 = random.randint(0, w - self.crop_size[0])
        y1 = random.randint(0, h - self.crop_size[1])
        img = img.crop((x1, y1, x1 + self.crop_size[0], y1 + self.crop_size[1]))
        seg = seg.crop((x1, y1, x1 + self.crop_size[0], y1 + self.crop_size[1]))
        return img, seg

class RandomFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img, seg):
        if random.random() < self.prob:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            seg = seg.transpose(Image.FLIP_LEFT_RIGHT)
        return img, seg

class PhotoMetricDistortion(object):
    def __call__(self, img, seg):
        img = np.array(img)
        brightness = random.uniform(0.6, 1.4)
        img = cv2.convertScaleAbs(img, alpha=brightness, beta=0)
        contrast = random.uniform(0.6, 1.4)
        img = cv2.convertScaleAbs(img, alpha=contrast, beta=0)
        saturation = random.uniform(0.6, 1.4)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        img_hsv[:,:,1] = img_hsv[:,:,1] * saturation
        img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
        return Image.fromarray(img), seg

class GenerateEdge(object):
    def __init__(self, edge_width=4):
        self.edge_width = edge_width

    def __call__(self, seg):
        seg_np = np.array(seg)
        # 确保 seg_np 是 uint8 类型
        if seg_np.dtype != np.uint8:
            unique_values = np.unique(seg_np)
            if np.any(seg_np > 255) or np.any(seg_np < 0):
                print(f"Warning: Invalid label values found in seg_np: {unique_values}")
            seg_np = seg_np.astype(np.uint8)
        edge = cv2.Canny(seg_np, 100, 200) / 255.0
        kernel = np.ones((self.edge_width * 2 + 1, self.edge_width * 2 + 1), np.uint8)
        edge = cv2.dilate(edge, kernel)
        return edge