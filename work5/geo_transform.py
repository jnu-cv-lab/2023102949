import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取本地自己的图片
img = cv2.imread("test.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
height, width = img.shape[:2]

# 1. 相似变换：旋转+等比缩放
center = (width // 2, height // 2)
mat_similar = cv2.getRotationMatrix2D(center, 25, 0.85)
img_similar = cv2.warpAffine(img, mat_similar, (width, height))

# 2. 仿射变换
pts1 = np.float32([[60,60], [260,60], [60,260]])
pts2 = np.float32([[40,100], [280,70], [80,280]])
mat_affine = cv2.getAffineTransform(pts1, pts2)
img_affine = cv2.warpAffine(img, mat_affine, (width, height))

# 3. 透视变换
src_pts = np.float32([[0,0], [width,0], [0,height], [width,height]])
dst_pts = np.float32([[40,30], [width-60,50], [30,height-40], [width-40,height-60]])
mat_per = cv2.getPerspectiveTransform(src_pts, dst_pts)
img_per = cv2.warpPerspective(img, mat_per, (width, height))

# 统一画图展示四张对比图
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.figure(figsize=(14,8))

plt.subplot(2,2,1)
plt.imshow(img)
plt.title("原始图像")
plt.axis("off")

plt.subplot(2,2,2)
plt.imshow(img_similar)
plt.title("相似变换")
plt.axis("off")

plt.subplot(2,2,3)
plt.imshow(img_affine)
plt.title("仿射变换")
plt.axis("off")

plt.subplot(2,2,4)
plt.imshow(img_per)
plt.title("透视变换")
plt.axis("off")

plt.tight_layout()
# 自动保存图片到文件夹
plt.savefig("实验结果图.png", dpi=150, bbox_inches='tight')
plt.show()
# ======================== A4 透视畸变校正（直接运行） ========================
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取你拍的图片
img = cv2.imread("paper.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ✅ 这是我专门给你算的坐标！
pts_src = np.float32([
    [165, 339],   # 左上
    [853, 361],   # 右上
    [253, 1495],  # 左下
    [1150, 1313]  # 右下
])

# 目标标准A4大小
w, h = 600, 800
pts_dst = np.float32([[0, 0], [w, 0], [0, h], [w, h]])

# 透视变换
M = cv2.getPerspectiveTransform(pts_src, pts_dst)
img_corrected = cv2.warpPerspective(img, M, (w, h))

# 显示 + 保存
plt.figure(figsize=(12, 6))
plt.subplot(121), plt.imshow(img), plt.title("畸变原图"), plt.axis('off')
plt.subplot(122), plt.imshow(img_corrected), plt.title("校正后"), plt.axis('off')
plt.savefig("A4校正结果.png", dpi=150, bbox_inches='tight')
plt.show()