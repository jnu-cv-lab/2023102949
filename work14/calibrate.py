import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt


# =========================
# 1. 参数设置
# =========================

# 棋盘格内角点数量：列数 × 行数
# 作业推荐为 9 × 6
CHECKERBOARD = (9, 6)

# 方格真实边长，单位 mm
# 按你的实际棋盘格填写，例如 25.0 或 30.0
SQUARE_SIZE = 22.0

# 图片文件夹
IMAGE_DIR = "images"

# 输出文件夹
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# 2. 准备三维世界坐标点
# =========================

cols, rows = CHECKERBOARD

# objp 的形状为 54 × 3
# 每个点是棋盘格坐标系下的三维点，Z = 0
objp = np.zeros((rows * cols, 3), np.float32)
objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)

# 乘以真实方格边长，得到真实物理坐标，单位 mm
objp = objp * SQUARE_SIZE

# 存储所有图片的三维点和二维角点
objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane

# 亚像素角点优化终止条件
criteria = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001
)


# =========================
# 3. 读取所有图片
# =========================

image_paths = []
for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
    image_paths.extend(glob.glob(os.path.join(IMAGE_DIR, ext)))

image_paths = sorted(image_paths)

if len(image_paths) == 0:
    print("错误：images 文件夹中没有找到图片。")
    exit()

print(f"共读取到 {len(image_paths)} 张图片。")


# =========================
# 4. 检测棋盘格角点
# =========================

valid_images = []
image_size = None

for idx, fname in enumerate(image_paths):
    img = cv2.imread(fname)

    if img is None:
        print(f"无法读取图片：{fname}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image_size = gray.shape[::-1]  # width, height

    # 检测棋盘格角点
    ret, corners = cv2.findChessboardCorners(
        gray,
        CHECKERBOARD,
        cv2.CALIB_CB_ADAPTIVE_THRESH
        + cv2.CALIB_CB_NORMALIZE_IMAGE
        + cv2.CALIB_CB_FAST_CHECK
    )

    if ret:
        objpoints.append(objp)

        # 亚像素精度优化
        corners_subpix = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            criteria
        )

        imgpoints.append(corners_subpix)
        valid_images.append(fname)

        # 绘制角点检测结果
        corner_img = img.copy()
        cv2.drawChessboardCorners(
            corner_img,
            CHECKERBOARD,
            corners_subpix,
            ret
        )

        save_name = os.path.join(
            OUTPUT_DIR,
            f"corners_{idx + 1:02d}.jpg"
        )
        cv2.imwrite(save_name, corner_img)

        print(f"[成功] 检测到角点：{fname}")

    else:
        print(f"[失败] 未检测到角点：{fname}")


print("\n==============================")
print(f"有效标定图片数量：{len(valid_images)}")
print("==============================\n")

if len(valid_images) < 10:
    print("警告：有效图片数量较少，建议至少 15 张，且姿态要丰富。")


# =========================
# 5. 相机标定
# =========================

if len(objpoints) == 0:
    print("错误：没有任何图片检测到棋盘格角点，无法标定。")
    exit()

ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints,
    imgpoints,
    image_size,
    None,
    None
)

print("相机标定完成。")
print("\n相机内参矩阵 K：")
print(camera_matrix)

print("\n畸变参数 D = [k1, k2, p1, p2, k3]：")
print(dist_coeffs.ravel())


# =========================
# 6. 计算平均重投影误差
# =========================

total_error = 0

for i in range(len(objpoints)):
    imgpoints_projected, _ = cv2.projectPoints(
        objpoints[i],
        rvecs[i],
        tvecs[i],
        camera_matrix,
        dist_coeffs
    )

    error = cv2.norm(
        imgpoints[i],
        imgpoints_projected,
        cv2.NORM_L2
    ) / len(imgpoints_projected)

    total_error += error

mean_error = total_error / len(objpoints)

print("\n平均重投影误差：")
print(f"{mean_error:.4f} pixels")


# =========================
# 7. 保存标定结果
# =========================

result_txt = os.path.join(OUTPUT_DIR, "calibration_results.txt")

with open(result_txt, "w", encoding="utf-8") as f:
    f.write("Camera Calibration Results\n")
    f.write("==========================\n\n")

    f.write(f"Checkerboard inner corners: {CHECKERBOARD[0]} x {CHECKERBOARD[1]}\n")
    f.write(f"Square size: {SQUARE_SIZE} mm\n")
    f.write(f"Image size: {image_size[0]} x {image_size[1]}\n")
    f.write(f"Valid images: {len(valid_images)}\n\n")

    f.write("Camera matrix K:\n")
    f.write(str(camera_matrix))
    f.write("\n\n")

    f.write("Distortion coefficients D = [k1, k2, p1, p2, k3]:\n")
    f.write(str(dist_coeffs.ravel()))
    f.write("\n\n")

    f.write(f"Mean reprojection error: {mean_error:.4f} pixels\n\n")

    f.write("Valid images:\n")
    for img_name in valid_images:
        f.write(img_name + "\n")

np.savez(
    os.path.join(OUTPUT_DIR, "calibration_data.npz"),
    camera_matrix=camera_matrix,
    dist_coeffs=dist_coeffs,
    rvecs=rvecs,
    tvecs=tvecs,
    mean_error=mean_error
)

print(f"\n标定结果已保存到：{result_txt}")


# =========================
# 8. 去畸变处理
# =========================

test_img_path = valid_images[0]
img = cv2.imread(test_img_path)
h, w = img.shape[:2]

new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
    camera_matrix,
    dist_coeffs,
    (w, h),
    1,
    (w, h)
)

undistorted = cv2.undistort(
    img,
    camera_matrix,
    dist_coeffs,
    None,
    new_camera_matrix
)

# 保存去畸变图
cv2.imwrite(os.path.join(OUTPUT_DIR, "original.jpg"), img)
cv2.imwrite(os.path.join(OUTPUT_DIR, "undistorted.jpg"), undistorted)

# 拼接原图和去畸变图，方便写报告
comparison = np.hstack((img, undistorted))
cv2.imwrite(os.path.join(OUTPUT_DIR, "comparison.jpg"), comparison)

print("去畸变图像已保存：")
print("output/original.jpg")
print("output/undistorted.jpg")
print("output/comparison.jpg")


# =========================
# 9. 使用 matplotlib 保存对比图
# =========================

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
undistorted_rgb = cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(undistorted_rgb)
plt.title("Undistorted Image")
plt.axis("off")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "undistortion_comparison.png"), dpi=300)
plt.close()

print("matplotlib 对比图已保存：output/undistortion_comparison.png")