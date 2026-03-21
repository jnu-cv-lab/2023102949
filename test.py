import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---------------------- 任务1：使用 OpenCV 读取一张测试图片 ----------------------
# 把图片放到和 test.py 同一个文件夹，替换成你的图片名，比如 test.jpg
img = cv2.imread("test.jpg.png")

if img is None:
    print("❌ 图片读取失败，请检查图片路径和文件名！")
else:
    print("✅ 图片读取成功！")

    # ---------------------- 任务2：输出图像基本信息 ----------------------
    height, width, channels = img.shape
    dtype = img.dtype
    print(f"📊 图像尺寸：宽度 = {width}, 高度 = {height}")
    print(f"🎨 通道数：{channels}（彩色图为3，灰度图为1）")
    print(f"📝 数据类型：{dtype}")

    # ---------------------- 任务3：显示原图 ----------------------
    # 方法1：OpenCV 显示（推荐）
    cv2.imshow("Original Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 方法2：Matplotlib 显示（颜色会从 BGR 转 RGB）
    # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # plt.imshow(img_rgb)
    # plt.axis("off")
    # plt.title("Original Image")
    # plt.show()

    # ---------------------- 任务4：转换为灰度图并显示 ----------------------
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Grayscale Image", gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # ---------------------- 任务5：保存灰度图为新文件 ----------------------
    cv2.imwrite("gray_test.jpg", gray)
    print("💾 灰度图已保存为：gray_test.jpg")

    # ---------------------- 任务6：NumPy 简单操作 ----------------------
    # 示例1：输出某个像素值（坐标 (100, 100)）
    pixel_value = img[100, 100]
    print(f"📍 像素 (100,100) 的 BGR 值：{pixel_value}")

    # 示例2：裁剪左上角 100x100 区域并保存
    crop_img = img[0:100, 0:100]  # 切片 [y_start:y_end, x_start:x_end]
    cv2.imwrite("crop_test.jpg", crop_img)
    print("✂️ 左上角裁剪图已保存为：crop_test.jpg")