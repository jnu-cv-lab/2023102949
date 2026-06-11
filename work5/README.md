# 2023102949-automation-lizhen-wor# 图像几何变换与透视校正实验报告

## 一、实验目的
1. 理解相似变换、仿射变换、透视变换的数学原理与几何特性。
2. 掌握 OpenCV 中 warpAffine、warpPerspective 等核心函数的使用方法。
3. 实现斜拍 A4 纸的透视畸变校正，验证投影变换的实际应用价值。
4. 对比三种变换对图像几何性质的影响差异。

## 二、实验环境
- 操作系统：Windows 10 + WSL2
- 编程语言：Python 3.x
- 依赖库：OpenCV-Python、NumPy、Matplotlib

## 三、实验原理
1. 相似变换：保持形状、旋转、缩放和平移，几何性质不变。
2. 仿射变换：保持直线和平行，不保持垂直与形状。
3. 透视变换：仅保持直线，可校正透视畸变，需要四点定位。

## 四、实验内容
1. 对测试图像实现相似变换、仿射变换、透视变换。
2. 对比三种变换对图像结构的影响。
3. 对斜拍 A4 纸进行四点透视校正。
4. 保存并分析实验结果。

## 五、核心代码
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

desktop = "/mnt/c/Users/lz/Desktop"

img = cv2.imread("test.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h, w = img.shape[:2]

center = (w//2, h//2)
M_similar = cv2.getRotationMatrix2D(center, 25, 0.85)
img_similar = cv2.warpAffine(img, M_similar, (w, h))

pts1 = np.float32([[60,60],[260,60],[60,260]])
pts2 = np.float32([[40,100],[280,70],[80,280]])
M_affine = cv2.getAffineTransform(pts1, pts2)
img_affine = cv2.warpAffine(img, M_affine, (w, h))

src = np.float32([[0,0],[w,0],[0,h],[w,h]])
dst = np.float32([[40,30],[w-60,50],[30,h-40],[w-40,h-60]])
M_per = cv2.getPerspectiveTransform(src, dst)
img_per = cv2.warpPerspective(img, M_per, (w, h))

plt.figure(figsize=(14,8))
plt.subplot(221),plt.imshow(img),plt.title("Original"),plt.axis('off')
plt.subplot(222),plt.imshow(img_similar),plt.title("Similarity"),plt.axis('off')
plt.subplot(223),plt.imshow(img_affine),plt.title("Affine"),plt.axis('off')
plt.subplot(224),plt.imshow(img_per),plt.title("Perspective"),plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(desktop, "transform_result.png"), dpi=150, bbox_inches='tight')
plt.close()

paper = cv2.imread("paper.jpg")
paper = cv2.cvtColor(paper, cv2.COLOR_BGR2RGB)

pts_src = np.float32([[220,320],[1050,330],[180,1580],[1100,1590]])
tw, th = 600, 850
pts_dst = np.float32([[0,0],[tw,0],[0,th],[tw,th]])

M = cv2.getPerspectiveTransform(pts_src, pts_dst)
corrected = cv2.warpPerspective(paper, M, (tw, th))

plt.figure(figsize=(12,6))
plt.subplot(121),plt.imshow(paper),plt.title("Distorted"),plt.axis('off')
plt.subplot(122),plt.imshow(corrected),plt.title("Corrected"),plt.axis('off')
plt.savefig(os.path.join(desktop, "a4_corrected.png"), dpi=150, bbox_inches='tight')
plt.close()

print("所有结果已保存到桌面")k5
