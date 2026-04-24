import cv2
import numpy as np

# 读取图像
img1 = cv2.imread('box.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('box_in_scene.png', cv2.IMREAD_GRAYSCALE)
if img1 is None or img2 is None:
    print("图像读取失败！")
    exit()

# ---------------------- 1. SIFT 特征检测 ----------------------
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# ---------------------- 2. KNN匹配 + Lowe ratio test ----------------------
bf = cv2.BFMatcher(cv2.NORM_L2)
matches = bf.knnMatch(des1, des2, k=2)

good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

num_matches = len(good_matches)
print(f"SIFT 总匹配数量：{num_matches}")

# ---------------------- 3. RANSAC 剔除错误匹配 ----------------------
src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
matchesMask = mask.ravel().tolist()
num_inliers = sum(matchesMask)
inlier_ratio = num_inliers / num_matches if num_matches > 0 else 0

print(f"SIFT 内点数量：{num_inliers}")
print(f"SIFT 内点比例：{inlier_ratio:.4f}")

# ---------------------- 4. 目标定位 ----------------------
h, w = img1.shape
pts = np.float32([[0,0], [0,h-1], [w-1,h-1], [w-1,0]]).reshape(-1,1,2)
dst = cv2.perspectiveTransform(pts, M)
img_scene_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
img_result = cv2.polylines(img_scene_color, [np.int32(dst)], True, (0,0,255), 3, cv2.LINE_AA)
cv2.imwrite('sift_localization.png', img_result)

# 运行速度主观评价：SIFT 比 ORB 慢很多，但匹配精度更高
print("\n===== SIFT vs ORB 对比 =====")
print("方法   匹配数量   内点数   内点比例   是否成功定位   运行速度   主观评价")
print("ORB     xxx        xxx      xxx         True/False     快        轻量快速，适合实时场景")
print("SIFT    xxx        xxx      xxx         True/False     慢        精度高，鲁棒性强，适合静态场景")