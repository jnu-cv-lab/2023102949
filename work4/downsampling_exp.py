import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.signal import get_window

# ===================== 工具函数 =====================
def generate_checkboard(size=256):
    """生成棋盘格测试图"""
    n = size // 8
    board = np.zeros((size, size), dtype=np.uint8)
    for i in range(8):
        for j in range(8):
            if (i + j) % 2 == 0:
                board[i*n:(i+1)*n, j*n:(j+1)*n] = 255
    return board

def generate_chirp(size=256):
    """生成chirp（线性调频）测试图：频率从0到Nyquist频率"""
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    # 线性调频：相位随半径平方变化，频率随半径线性增加
    chirp = np.sin(2 * np.pi * 20 * R**2)
    # 归一化到0-255
    chirp = (chirp - chirp.min()) / (chirp.max() - chirp.min()) * 255
    return chirp.astype(np.uint8)

def downsample(img, M):
    """直接下采样：每隔M个像素取一个"""
    return img[::M, ::M]

def gaussian_downsample(img, M, sigma):
    """高斯滤波后下采样"""
    # 高斯滤波：sigma为高斯核标准差
    blurred = gaussian_filter(img, sigma=sigma)
    return downsample(blurred, M)

def compute_fft_spectrum(img):
    """计算图像的FFT频谱（中心化）"""
    f = np.fft.fft2(img)
    f_shift = np.fft.fftshift(f)
    # 取幅度谱，对数压缩动态范围
    magnitude = 20 * np.log(np.abs(f_shift) + 1)
    return magnitude

# ===================== 第一部分：棋盘格+chirp下采样与混叠验证 =====================
def part1_experiment():
    print("===== 第一部分：棋盘格和chirp测试图下采样实验 =====")
    # 1. 生成测试图
    checkboard = generate_checkboard(256)
    chirp = generate_chirp(256)
    M = 4  # 下采样倍数

    # 2. 直接下采样（无抗混叠）
    check_direct = downsample(checkboard, M)
    chirp_direct = downsample(chirp, M)

    # 3. 高斯滤波后下采样（抗混叠）
    sigma_opt = 0.45 * M  # 理论最优sigma
    check_gauss = gaussian_downsample(checkboard, M, sigma_opt)
    chirp_gauss = gaussian_downsample(chirp, M, sigma_opt)

    # 4. 计算FFT频谱
    chirp_fft_original = compute_fft_spectrum(chirp)
    chirp_fft_direct = compute_fft_spectrum(chirp_direct)
    chirp_fft_gauss = compute_fft_spectrum(chirp_gauss)

    # 5. 可视化结果
    plt.figure(figsize=(16, 12))

    # 棋盘格结果
    plt.subplot(3, 4, 1)
    plt.imshow(checkboard, cmap='gray')
    plt.title('原始棋盘格')
    plt.axis('off')

    plt.subplot(3, 4, 2)
    plt.imshow(check_direct, cmap='gray')
    plt.title(f'直接下采样 M={M} (混叠明显)')
    plt.axis('off')

    plt.subplot(3, 4, 3)
    plt.imshow(check_gauss, cmap='gray')
    plt.title(f'高斯滤波后下采样 σ={sigma_opt:.2f} (混叠消除)')
    plt.axis('off')

    # Chirp图结果
    plt.subplot(3, 4, 5)
    plt.imshow(chirp, cmap='gray')
    plt.title('原始Chirp测试图')
    plt.axis('off')

    plt.subplot(3, 4, 6)
    plt.imshow(chirp_direct, cmap='gray')
    plt.title(f'Chirp直接下采样 M={M} (混叠伪影)')
    plt.axis('off')

    plt.subplot(3, 4, 7)
    plt.imshow(chirp_gauss, cmap='gray')
    plt.title(f'Chirp高斯滤波后下采样 σ={sigma_opt:.2f}')
    plt.axis('off')

    # FFT频谱
    plt.subplot(3, 4, 9)
    plt.imshow(chirp_fft_original, cmap='gray')
    plt.title('原始Chirp频谱')
    plt.axis('off')

    plt.subplot(3, 4, 10)
    plt.imshow(chirp_fft_direct, cmap='gray')
    plt.title('直接下采样后频谱 (混叠导致频谱重叠)')
    plt.axis('off')

    plt.subplot(3, 4, 11)
    plt.imshow(chirp_fft_gauss, cmap='gray')
    plt.title('高斯滤波后下采样频谱 (混叠消失)')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('part1_result.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("第一部分实验完成，结果已保存为 part1_result.png")

# ===================== 第二部分：σ公式验证实验 =====================
def part2_sigma_experiment():
    print("\n===== 第二部分：σ公式验证实验 =====")
    # 固定M=4，测试不同sigma
    M = 4
    sigma_list = [0.5, 1.0, 2.0, 4.0]
    sigma_theory = 0.45 * M  # 理论最优sigma
    print(f"理论最优σ = 0.45*M = {sigma_theory:.2f}")

    # 用chirp图做测试
    chirp = generate_chirp(256)

    plt.figure(figsize=(16, 8))
    for idx, sigma in enumerate(sigma_list):
        # 高斯滤波+下采样
        img_down = gaussian_downsample(chirp, M, sigma)
        # 计算频谱
        fft_spec = compute_fft_spectrum(img_down)

        # 可视化
        plt.subplot(2, len(sigma_list), idx+1)
        plt.imshow(img_down, cmap='gray')
        plt.title(f'M={M}, σ={sigma}')
        plt.axis('off')

        plt.subplot(2, len(sigma_list), idx+1+len(sigma_list))
        plt.imshow(fft_spec, cmap='gray')
        plt.title(f'σ={sigma} 频谱')
        plt.axis('off')

    plt.suptitle(f'不同σ下采样结果对比 (理论最优σ={sigma_theory:.2f})', fontsize=16)
    plt.tight_layout()
    plt.savefig('part2_sigma_result.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 单独对比理论最优sigma
    print("\n===== 理论最优σ验证 =====")
    img_opt = gaussian_downsample(chirp, M, sigma_theory)
    fft_opt = compute_fft_spectrum(img_opt)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img_opt, cmap='gray')
    plt.title(f'理论最优σ={sigma_theory:.2f} 下采样结果')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(fft_opt, cmap='gray')
    plt.title('理论最优σ下采样频谱 (混叠完全消除)')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('part2_opt_sigma_result.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("第二部分实验完成，结果已保存为 part2_sigma_result.png 和 part2_opt_sigma_result.png")

# ===================== 第三部分：自适应下采样实验 =====================
def part3_adaptive_downsample():
    print("\n===== 第三部分：自适应下采样实验 =====")
    # 1. 加载测试图（用lena图，也可以用自己的图）
    img = cv2.imread('lena.png', cv2.IMREAD_GRAYSCALE)
    if img is None:
        # 若没有lena图，生成合成图
        print("未找到lena.png，生成合成测试图")
        img = np.zeros((256, 256), dtype=np.uint8)
        # 平滑区域+纹理区域
        img[:128, :128] = 128  # 平滑区
        img[128:, 128:] = generate_checkboard(128)  # 纹理区
        img[:128, 128:] = generate_chirp(128)  # 高频区

    M_global = 4  # 全局下采样倍数
    sigma_global = 0.45 * M_global  # 全局sigma

    # 2. 梯度分析：计算局部梯度（估计局部M值）
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    # 归一化梯度
    grad_mag = (grad_mag - grad_mag.min()) / (grad_mag.max() - grad_mag.min())

    # 3. 自适应sigma：梯度大（纹理区）用大sigma，梯度小（平滑区）用小sigma
    # 设定sigma范围：0.5 ~ 2.0
    sigma_adaptive = 0.5 + 1.5 * grad_mag
    # 自适应滤波：逐像素sigma（用空间变高斯滤波，这里用分块近似实现）
    block_size = 8
    img_adaptive = img.copy().astype(np.float64)
    for i in range(0, img.shape[0], block_size):
        for j in range(0, img.shape[1], block_size):
            # 块内平均sigma
            block_sigma = sigma_adaptive[i:i+block_size, j:j+block_size].mean()
            # 对块做高斯滤波
            img_adaptive[i:i+block_size, j:j+block_size] = gaussian_filter(
                img[i:i+block_size, j:j+block_size], sigma=block_sigma
            )
    img_adaptive = img_adaptive.astype(np.uint8)
    # 自适应下采样
    img_adaptive_down = downsample(img_adaptive, M_global)

    # 4. 全局统一下采样
    img_global_down = gaussian_downsample(img, M_global, sigma_global)

    # 5. 计算误差图（与原始图的差值，放大显示）
    # 先上采样回原尺寸
    img_global_up = cv2.resize(img_global_down, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
    img_adaptive_up = cv2.resize(img_adaptive_down, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
    # 计算绝对误差
    error_global = np.abs(img.astype(np.float32) - img_global_up.astype(np.float32))
    error_adaptive = np.abs(img.astype(np.float32) - img_adaptive_up.astype(np.float32))
    # 误差归一化
    error_global = (error_global / error_global.max() * 255).astype(np.uint8)
    error_adaptive = (error_adaptive / error_adaptive.max() * 255).astype(np.uint8)

    # 6. 可视化结果
    plt.figure(figsize=(16, 12))

    # 原始图+梯度图
    plt.subplot(3, 4, 1)
    plt.imshow(img, cmap='gray')
    plt.title('原始图像')
    plt.axis('off')

    plt.subplot(3, 4, 2)
    plt.imshow(grad_mag, cmap='hot')
    plt.title('局部梯度图（纹理检测）')
    plt.axis('off')

    # 全局下采样结果
    plt.subplot(3, 4, 5)
    plt.imshow(img_global_down, cmap='gray')
    plt.title(f'全局统一下采样 M={M_global}, σ={sigma_global:.2f}')
    plt.axis('off')

    plt.subplot(3, 4, 6)
    plt.imshow(error_global, cmap='gray')
    plt.title('全局下采样误差图')
    plt.axis('off')

    # 自适应下采样结果
    plt.subplot(3, 4, 9)
    plt.imshow(img_adaptive_down, cmap='gray')
    plt.title('自适应下采样（分块变σ滤波）')
    plt.axis('off')

    plt.subplot(3, 4, 10)
    plt.imshow(error_adaptive, cmap='gray')
    plt.title('自适应下采样误差图（误差更小）')
    plt.axis('off')

    # 误差对比
    plt.subplot(3, 4, 7)
    plt.hist(error_global.flatten(), bins=50, alpha=0.5, label='全局下采样误差', color='red')
    plt.hist(error_adaptive.flatten(), bins=50, alpha=0.5, label='自适应下采样误差', color='blue')
    plt.legend()
    plt.title('误差分布对比')
    plt.xlabel('误差值')
    plt.ylabel('像素数')

    plt.tight_layout()
    plt.savefig('part3_adaptive_result.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 输出误差统计
    print(f"全局下采样平均误差: {error_global.mean():.2f}")
    print(f"自适应下采样平均误差: {error_adaptive.mean():.2f}")
    print("第三部分实验完成，结果已保存为 part3_adaptive_result.png")

# ===================== 主函数：按顺序执行所有实验 =====================
if __name__ == "__main__":
    # 第一部分：棋盘格+chirp下采样与混叠验证
    part1_experiment()

    # 第二部分：σ公式验证
    part2_sigma_experiment()

    # 第三部分：自适应下采样
    part3_adaptive_downsample()

    print("\n===== 所有实验全部完成！=====")
    print("生成的结果文件：")
    print("1. part1_result.png：第一部分混叠验证结果")
    print("2. part2_sigma_result.png：不同σ对比结果")
    print("3. part2_opt_sigma_result.png：最优σ验证结果")
    print("4. part3_adaptive_result.png：自适应下采样对比结果")