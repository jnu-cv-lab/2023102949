# 计算机视觉实验一：Python视觉开发环境搭建与图像基本读写

## 实验目的
1.  掌握使用 OpenCV 读取、显示和保存图像的基本方法，理解图像读取失败的排查方式。
2.  学会获取图像核心信息（尺寸、通道数、数据类型），理解彩色图与灰度图的通道差异。
3.  掌握彩色图像转灰度图像的实现方法，理解 OpenCV 中 BGR 色彩空间特性。
4.  熟悉 NumPy 对图像数组的基础操作（像素值获取、区域裁剪），理解图像的数组存储形式。
5.  完成 Python 视觉开发环境搭建，掌握依赖库管理与项目结构组织。

## 实验环境
| 配置项         | 具体信息                                                                 |
|----------------|--------------------------------------------------------------------------|
| 开发语言       | Python 3.x                                                               |
| 核心依赖库     | opencv-python==4.13.0.92、numpy==2.4.3、matplotlib==3.10.8                |
| 依赖管理文件   | `requirements-basic.txt`                                                 |
| 开发工具       | VS Code / PyCharm                                                         |
| 实验代码文件   | `code/test.py`                                                            |

## 实验步骤
### 1. 环境搭建
1.  项目目录结构：
    ```
    ├── requirements-basic.txt  # 依赖库版本管理
    ├── README.md               # 项目说明与实验报告
    └── code/
        ├── test.py             # 实验核心代码
        └── test.jpg.png        # 测试原始图片
    ```
2.  安装依赖：
    ```bash
    pip install -r requirements-basic.txt
    ```

### 2. 核心代码实现
```python
import cv2
import numpy as np

# 任务1：读取测试图片
img = cv2.imread("test.jpg.png")
if img is None:
    print("❌ 图片读取失败，请检查路径与文件名！")
else:
    print("✅ 图片读取成功！")

    # 任务2：输出图像基本信息
    height, width, channels = img.shape
    dtype = img.dtype
    print(f"📊 图像尺寸：宽度 = {width}, 高度 = {height}")
    print(f"🎨 通道数：{channels}（彩色图为3，灰度图为1）")
    print(f"📝 数据类型：{dtype}")

    # 任务3：显示原图
    cv2.imshow("Original Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 任务4：转换为灰度图并显示
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Grayscale Image", gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 任务5：保存灰度图
    cv2.imwrite("gray_test.jpg", gray)
    print("💾 灰度图已保存为：gray_test.jpg")

    # 任务6：NumPy 基础操作（获取像素值 + 裁剪图像）
    pixel_value = img[100, 100]  # 获取坐标(100,100)的BGR像素值
    print(f"📍 像素 (100,100) 的 BGR 值：{pixel_value}")

    crop_img = img[0:100, 0:100]  # 裁剪左上角100x100区域
    cv2.imwrite("crop_test.jpg", crop_img)
    print("✂️ 左上角裁剪图已保存为：crop_test.jpg")
```

### 3. 运行与验证
1.  进入代码目录：
    ```bash
    cd code
    ```
2.  执行代码：
    ```bash
    python test.py
    ```
3.  验证输出文件：`code/` 目录下生成 `gray_test.jpg`（灰度图）和 `crop_test.jpg`（裁剪图）。

## 实验结果
### 终端输出示例
```
✅ 图片读取成功！
📊 图像尺寸：宽度 = 1920, 高度 = 1080
🎨 通道数：3（彩色图为3，灰度图为1）
📝 数据类型：uint8
💾 灰度图已保存为：gray_test.jpg
📍 像素 (100,100) 的 BGR 值：[102 117 133]
✂️ 左上角裁剪图已保存为：crop_test.jpg
```

### 生成文件
| 文件名          | 功能说明                     |
|-----------------|------------------------------|
| `test.jpg.png`  | 原始测试图片                 |
| `gray_test.jpg` | 彩色图转换后的灰度图         |
| `crop_test.jpg` | 原始图左上角 100×100 裁剪图  |

## 问题与解决
1.  **图片读取失败**
    - 原因：图片路径/文件名与代码不一致，或文件损坏。
    - 解决：确保图片与 `test.py` 同目录，核对文件名完全一致。
2.  **缺少 `cv2` 模块**
    - 原因：未安装 OpenCV 库。
    - 解决：执行 `pip install -r requirements-basic.txt` 重新安装依赖。
3.  **Matplotlib 显示色彩失真**
    - 原因：OpenCV 读取为 BGR 格式，Matplotlib 默认显示 RGB。
    - 解决：添加 `img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` 转换后再显示。

## 实验总结
本次实验完成了 Python 视觉开发环境搭建与图像基础操作，掌握了 OpenCV 读取、显示、转换、保存图像的完整流程，理解了图像作为 NumPy 数组的存储形式与坐标规则。通过 `requirements-basic.txt` 实现了依赖管理，提升了项目可移植性。实验中体会到细节把控（如路径、色彩空间、坐标顺序）的重要性，为后续更复杂的计算机视觉任务打下了基础。
