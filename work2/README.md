# 2023102949-automation-# OpenCV 图像处理实验报告
> 基于 OpenCV 4 + C++ 完成图像读取、灰度转换、像素操作与区域裁剪实验

---

## 一、实验概述
### 1.1 实验要求
使用 OpenCV 库，用 C++ 编写程序完成以下 6 项任务：
1.  **任务1**：使用 OpenCV 读取一张测试图片
2.  **任务2**：输出图像基本信息（尺寸、通道数、像素类型等）
3.  **任务3**：显示原图
4.  **任务4**：将彩色图转换为灰度图并显示
5.  **任务5**：保存灰度图为新文件
6.  **任务6**：像素操作（输出像素值）/ 裁剪图像左上角区域并保存

### 1.2 开发环境
| 环境/工具 | 版本/说明 |
|-----------|-----------|
| 操作系统 | Windows 11 + WSL Ubuntu 24.04 |
| 开发工具 | VS Code（WSL 远程开发） |
| 编程语言 | C++ 17 |
| 图像处理库 | OpenCV 4.6.0 |
| 编译器 | g++ 12.1.0 |

---

## 二、实验原理
### 2.1 核心原理说明
1.  **图像读取**：`cv::imread()` 函数加载图像，默认以 BGR 格式存储彩色图像
2.  **图像属性获取**：通过 `cv::Mat` 类的 `cols`（宽度）、`rows`（高度）、`channels()`（通道数）、`total()`（总像素数）获取图像信息
3.  **灰度转换**：`cv::cvtColor(COLOR_BGR2GRAY)` 将 3 通道 BGR 图转为 1 通道灰度图
4.  **图像显示**：`cv::namedWindow()` 创建可缩放窗口，`cv::imshow()` 显示图像，`cv::waitKey(0)` 防止窗口闪退
5.  **图像保存**：`cv::imwrite()` 将处理后的图像写入本地文件
6.  **像素访问**：`cv::Mat::at<cv::Vec3b>(y, x)` 访问指定坐标的 BGR 像素值
7.  **区域裁剪**：`cv::Rect` 定义感兴趣区域（ROI），从原图中截取指定大小的子图

---

## 三、核心代码实现
### 3.1 主程序 `main.cpp`
```cpp
#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;

int main() {
    // ====================== 任务1：读取测试图片 ======================
    Mat img = imread("test.jpg");
    if (img.empty()) {
        cout << "❌ 错误：无法读取test.jpg，请检查文件是否在当前目录！" << endl;
        return -1;
    }
    cout << "✅ 任务1完成：成功读取测试图片" << endl << endl;

    // ====================== 任务2：输出图像基本信息 ======================
    cout << "===== 图像基本信息 =====" << endl;
    cout << "图像宽度：" << img.cols << " 像素" << endl;
    cout << "图像高度：" << img.rows << " 像素" << endl;
    cout << "通道数：" << img.channels() << endl;
    cout << "像素数据类型：" << img.type() << endl;
    cout << "总像素数：" << img.total() << endl << endl;
    cout << "✅ 任务2完成：已输出图像信息" << endl << endl;

    // ====================== 任务3：显示原图 ======================
    namedWindow("原图", WINDOW_NORMAL);
    imshow("原图", img);
    cout << "✅ 任务3完成：已显示原图" << endl << endl;

    // ====================== 任务4：转换为灰度图并显示 ======================
    Mat gray_img;
    cvtColor(img, gray_img, COLOR_BGR2GRAY);
    namedWindow("灰度图", WINDOW_NORMAL);
    imshow("灰度图", gray_img);
    cout << "✅ 任务4完成：已转换并显示灰度图" << endl << endl;

    // ====================== 任务5：保存灰度图 ======================
    imwrite("gray_output.jpg", gray_img);
    cout << "✅ 任务5完成：灰度图已保存为 gray_output.jpg" << endl << endl;

    // ====================== 任务6：像素操作 + 裁剪 ======================
    cout << "===== 任务6：像素与裁剪操作 =====" << endl;
    // 输出左上角(0,0)的BGR像素值
    Vec3b pixel = img.at<Vec3b>(0, 0);
    cout << "左上角(0,0)像素值（BGR格式）：" << endl;
    cout << "B(蓝)：" << (int)pixel[0] << endl;
    cout << "G(绿)：" << (int)pixel[1] << endl;
    cout << "R(红)：" << (int)pixel[2] << endl << endl;

    // 裁剪左上角100x100区域并保存
    Mat roi = img(Rect(0, 0, 100, 100));
    imwrite("roi.jpg", roi);
    namedWindow("裁剪区域", WINDOW_NORMAL);
    imshow("裁剪区域", roi);
    cout << "✅ 任务6完成：左上角100x100区域已保存为 roi.jpg" << endl << endl;

    // 等待按键，关闭所有窗口
    waitKey(0);
    destroyAllWindows();

    cout << "🎉 所有实验任务全部完成！" << endl;
    return 0;
}
```

---

## 四、编译与运行
### 4.1 依赖安装
```bash
# 安装OpenCV开发库
sudo apt install libopencv-dev
# 安装编译工具
sudo apt install build-essential
```

### 4.2 编译命令
```bash
# 单文件编译（自动链接OpenCV）
g++ main.cpp -o main `pkg-config --cflags --libs opencv4`
```
> 注意：命令中的反引号（`）为键盘左上角 `Esc` 下方的按键，不可替换为单引号

### 4.3 运行命令
```bash
# 运行程序
./main
```

### 4.4 多文件编译说明（CMake 方式）
若需多文件编译，可创建 `CMakeLists.txt`：
```cmake
cmake_minimum_required(VERSION 3.10)
project(OpenCV_Image_Process)

# 查找OpenCV库
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# 生成可执行文件
add_executable(main main.cpp)
target_link_libraries(main ${OpenCV_LIBS})
```
编译步骤：
```bash
mkdir build && cd build
cmake ..
make
./main
```

---

## 五、实验结果
### 5.1 终端输出结果
```
✅ 任务1完成：成功读取测试图片

===== 图像基本信息 =====
图像宽度：2560 像素
图像高度：1440 像素
通道数：3
像素数据类型：16
总像素数：3686400

✅ 任务2完成：已输出图像信息

✅ 任务3完成：已显示原图

✅ 任务4完成：已转换并显示灰度图

✅ 任务5完成：灰度图已保存为 gray_output.jpg

===== 任务6：像素与裁剪操作 =====
左上角(0,0)像素值（BGR格式）：
B(蓝)：22
G(绿)：53
R(红)：88

✅ 任务6完成：左上角100x100区域已保存为 roi.jpg

🎉 所有实验任务全部完成！
```

### 5.2 生成文件说明
| 文件名 | 说明 |
|--------|------|
| `test.jpg` | 输入的测试原图 |
| `main` | 编译生成的可执行文件 |
| `gray_output.jpg` | 转换后的灰度图 |
| `roi.jpg` | 裁剪后的100×100区域图 |

### 5.3 图像显示结果
- 成功弹出3个窗口：**原图**、**灰度图**、**裁剪区域图**
- 所有窗口支持缩放，按任意键关闭窗口

---

## 六、实验总结
### 6.1 实验完成情况
✅ 全部6项实验任务均已完成，程序运行稳定，结果符合预期。

### 6.2 知识点掌握
1.  掌握了 OpenCV 图像读取、显示、保存的基础流程
2.  理解了彩色图像 BGR 通道的存储格式，学会了像素值的访问方法
3.  掌握了图像灰度化、区域裁剪的实现原理
4.  熟悉了 Linux 环境下 C++ 程序的编译、OpenCV 库的链接方法
5.  了解了多文件编译的 CMake 构建方式

### 6.3 问题与解决
1.  **问题1**：VS Code 提示 `opencv2/opencv.hpp` 头文件找不到
    - 解决：在 `c_cpp_properties.json` 中添加 `/usr/include/opencv4` 头文件路径
2.  **问题2**：编译报错 `expected '}' at end of input`
    - 解决：补全代码中缺失的闭合大括号，确保语法正确
3.  **问题3**：WSL 文件无法直接拖拽到 GitHub
    - 解决：通过 `cp` 命令将 WSL 文件复制到 Windows 桌面，再上传 GitHub

---

## 七、仓库文件结构
```
.
├── main.cpp          # 实验主程序
├── test.jpg          # 测试原图
├── gray_output.jpg   # 输出灰度图
├── roi.jpg           # 输出裁剪图
├── CMakeLists.txt    # 多文件编译配置（可选）
└── README.md         # 本实验报告
```

---


---

需要我帮你把这份报告再精简成**适合GitHub展示的短版README**，或者补充实验截图的说明吗？lizhen-work2
