import cv2
import numpy as np
import matplotlib.pyplot as plt

# ===================== 任务1：ORB 关键点检测 =====================
# 读取图片（绝对路径，不会找不到）
img1 = cv2.imread('/home/lz/ORB_Experiment/box.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('/home/lz/ORB_Experiment/box_in_scene.png', cv2.IMREAD_GRAYSCALE)

if img1 is None or img2 is None:
    print("图片读取失败！")
    exit()

# 初始化 ORB
orb = cv2.ORB_create(nfeatures=1000)
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

print("===== 任务1 结果 =====")
print(f"模板图关键点数量：{len(kp1)}")
print(f"场景图关键点数量：{len(kp2)}")
print(f"描述子维度：{des1.shape[1]}")

# 保存关键点图
img1_kp = cv2.drawKeypoints(img1, kp1, None, (0,255,0), 0)
img2_kp = cv2.drawKeypoints(img2, kp2, None, (0,255,0), 0)
cv2.imwrite('/home/lz/ORB_Experiment/box_keypoints.png', img1_kp)
cv2.imwrite('/home/lz/ORB_Experiment/scene_keypoints.png', img2_kp)

# ===================== 任务2：ORB 特征匹配 =====================
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

print("\n===== 任务2 结果 =====")
print(f"总匹配数量：{len(matches)}")

# 保存所有匹配图
img_match_all = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imwrite('/home/lz/ORB_Experiment/orb_matches.png', img_match_all)

# 保存前50最优匹配图
img_match_top50 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imwrite('/home/lz/ORB_Experiment/orb_matches_top50.png', img_match_top50)

# ===================== 任务3：RANSAC 剔除错误匹配 =====================
src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
matchesMask = mask.ravel().tolist()

num_matches = len(matches)
num_inliers = sum(matchesMask)
inlier_ratio = num_inliers / num_matches if num_matches > 0 else 0

print("\n===== 任务3 结果 =====")
print(f"总匹配数量：{num_matches}")
print(f"RANSAC 内点数量：{num_inliers}")
print(f"内点比例：{inlier_ratio:.4f}")
print("Homography 矩阵：\n", M)

# 保存 RANSAC 过滤图
draw_params = dict(matchColor=(0,255,0), matchesMask=matchesMask, flags=2)
img_ransac = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, **draw_params)
cv2.imwrite('/home/lz/ORB_Experiment/ransac_filtered_matches.png', img_ransac)

# ===================== 任务4：目标定位 =====================
h, w = img1.shape
pts = np.float32([[0,0], [0,h-1], [w-1,h-1], [w-1,0]]).reshape(-1,1,2)
dst = cv2.perspectiveTransform(pts, M)

# 画框
img_scene_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
img_result = cv2.polylines(img_scene_color, [np.int32(dst)], True, (0,0,255), 3, cv2.LINE_AA)
cv2.imwrite('/home/lz/ORB_Experiment/target_localization.png', img_result)

print("\n===== 任务4 结果 =====")
print("目标定位成功！已生成红色框选图")