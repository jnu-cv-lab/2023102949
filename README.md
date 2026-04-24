# ORB 特征检测、匹配与目标定位

## 实验目的
1. 掌握 ORB 特征点检测与描述子提取的基本原理与实现方法。
2. 掌握暴力匹配（BFMatcher）完成特征匹配的流程。
3. 理解 RANSAC 算法剔除误匹配、估计单应矩阵的作用。
4. 实现基于单应矩阵的目标定位与可视化。
5. 完成 `nfeatures` 参数对比实验，分析特征点数量对匹配效果的影响。
6. 对比 ORB 与 SIFT 算法的差异，理解不同局部特征的适用场景。

## 实验环境
| 配置项         | 具体信息                                                                 |
|----------------|--------------------------------------------------------------------------|
| 开发语言       | Python 3.x                                                               |
| 核心依赖库     | opencv-python、numpy、matplotlib                                          |
| 实验代码文件   | `main.py`、`compare_nfeatures.py`、`sift_experiment.py`                   |
| 实验图像文件   | `box.png`、`box_in_scene.png`                                            |

## 实验步骤
### 1. 项目目录结构
```
ORB_Experiment/
├── main.py                     # 任务1~4 主程序
├── compare_nfeatures.py        # 任务6 参数对比
├── sift_experiment.py          # SIFT 对比实验
├── box.png                     # 模板图像
├── box_in_scene.png            # 场景图像
├── box_keypoints.png
├── scene_keypoints.png
├── orb_matches.png
├── orb_matches_top50.png
├── ransac_filtered_matches.png
├── target_localization.png
└── sift_localization.png
```

### 2. 核心代码实现
#### 任务1~4：ORB 检测、匹配、RANSAC、目标定位
```python
import cv2
import numpy as np

img1 = cv2.imread('box.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('box_in_scene.png', cv2.IMREAD_GRAYSCALE)

orb = cv2.ORB_create(nfeatures=1000)
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

h, w = img1.shape
pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
dst = cv2.perspectiveTransform(pts, M)
img_result = cv2.polylines(cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR), [np.int32(dst)], True, (0,0,255), 3)
```

#### 任务6：nfeatures 参数对比
```python
def run_orb(nfeatures):
    orb = cv2.ORB_create(nfeatures=nfeatures)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    ...
```

#### SIFT 对比实验
```python
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
...
```

### 3. 运行方式
```bash
python main.py
python compare_nfeatures.py
python sift_experiment.py
```

## 实验结果
### 任务1：ORB 关键点检测
```
===== 任务1 结果 =====
模板图关键点数量：865
场景图关键点数量：1000
描述子维度：32
```

### 任务2：ORB 特征匹配
```
===== 任务2 结果 =====
总匹配数量：287
```

### 任务3：RANSAC 剔除误匹配
```
===== 任务3 结果 =====
总匹配数量：287
RANSAC 内点数量：52
内点比例：0.1812
```

### 任务4：目标定位
```
===== 任务4 结果 =====
目标定位成功！已生成红色框选图
```

### 任务6：nfeatures 参数对比
| nfeatures | 模板关键点 | 场景关键点 | 匹配数 | 内点数 | 内点比例 | 定位成功 |
|-----------|------------|------------|--------|--------|----------|----------|
| 500       | 453        | 500        | 148    | 32     | 0.2162   | True     |
| 1000      | 865        | 1000       | 287    | 52     | 0.1812   | True     |
| 2000      | 1589       | 1999       | 511    | 63     | 0.1233   | False    |

### SIFT 实验结果
```
===== SIFT 实验结果 =====
模板图关键点：443
场景图关键点：500
优质匹配对数：312
```

### 生成文件
| 文件名                        | 功能说明                     |
|-------------------------------|------------------------------|
| box_keypoints.png             | 模板图 ORB 关键点            |
| scene_keypoints.png           | 场景图 ORB 关键点            |
| orb_matches.png               | ORB 全部匹配                |
| orb_matches_top50.png         | ORB 前 50 最优匹配          |
| ransac_filtered_matches.png   | RANSAC 过滤匹配             |
| target_localization.png       | ORB 目标定位结果             |
| sift_localization.png         | SIFT 目标定位结果            |

## 问题与解决
1. **图像读取失败**
   - 原因：路径错误或文件名不匹配。
   - 解决：使用绝对路径或确保图片与代码同目录。

2. **变量未定义报错**
   - 原因：变量未提前赋值直接使用。
   - 解决：在使用前完成变量定义与赋值。

3. **匹配图重复/效果异常**
   - 原因：未正确排序或绘制逻辑重复。
   - 解决：按距离排序后再绘制前 N 个匹配。

## 实验总结
本次实验完整实现了 ORB 特征检测、特征匹配、RANSAC 误匹配剔除、单应矩阵估计与目标定位流程。通过参数对比实验发现，特征点数量并非越多越好，过多特征会引入噪声导致定位失败。SIFT 算法精度更高但速度较慢，ORB 则在速度与精度之间取得平衡。实验加深了对局部特征、匹配策略与几何变换的理解，为目标检测、图像拼接等高级视觉任务奠定基础。

