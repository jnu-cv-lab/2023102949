import cv2
import numpy as np

def run_orb_experiment(nfeatures):
    # 读取图像
    img1 = cv2.imread('box.png', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('box_in_scene.png', cv2.IMREAD_GRAYSCALE)
    if img1 is None or img2 is None:
        print("图像读取失败！")
        return None

    # ORB检测
    orb = cv2.ORB_create(nfeatures=nfeatures)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # 暴力匹配
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    num_matches = len(matches)

    # RANSAC
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()
    num_inliers = sum(matchesMask)
    inlier_ratio = num_inliers / num_matches if num_matches > 0 else 0

    # 定位是否成功（简单判断：边框是否在图像内）
    h, w = img1.shape
    pts = np.float32([[0,0], [0,h-1], [w-1,h-1], [w-1,0]]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts, M)
    loc_success = True
    for p in dst:
        if p[0][0] < 0 or p[0][0] > img2.shape[1] or p[0][1] < 0 or p[0][1] > img2.shape[0]:
            loc_success = False
            break

    return {
        "nfeatures": nfeatures,
        "kp1_num": len(kp1),
        "kp2_num": len(kp2),
        "num_matches": num_matches,
        "num_inliers": num_inliers,
        "inlier_ratio": inlier_ratio,
        "loc_success": loc_success
    }

# 测试三组参数
params = [500, 1000, 2000]
results = []
for p in params:
    res = run_orb_experiment(p)
    if res:
        results.append(res)

# 打印对比表格
print("===== 任务6 参数对比结果 =====")
print(f"{'nfeatures':<10} {'模板图关键点':<12} {'场景图关键点':<12} {'匹配数量':<8} {'内点数':<8} {'内点比例':<8} {'是否成功定位':<10}")
for res in results:
    print(f"{res['nfeatures']:<10} {res['kp1_num']:<12} {res['kp2_num']:<12} {res['num_matches']:<8} {res['num_inliers']:<8} {res['inlier_ratio']:.4f}      {res['loc_success']}")