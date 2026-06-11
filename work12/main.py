import numpy as np
def sinusoidal_pos_encoding(max_len: int, d_model: int) -> np.ndarray:
    # 初始化位置编码矩阵
    pe = np.zeros((max_len, d_model))
    # 所有位置下标
    pos = np.arange(0, max_len)[:, np.newaxis]
    # 计算公式里的分母项
    div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
    # 偶数维度 sin，奇数维度 cos
    pe[:, 0::2] = np.sin(pos * div_term)
    pe[:, 1::2] = np.cos(pos * div_term)
    return pe
def rotate_2d(x1: float, x2: float, theta: float) -> tuple[float, float]:
    # 计算余弦、正弦
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    # 二维旋转公式
    x1_rot = x1 * cos_t - x2 * sin_t
    x2_rot = x1 * sin_t + x2 * cos_t
    return x1_rot, x2_rot

def rope_encoding(x: np.ndarray, pos: int, d_model: int) -> np.ndarray:
    x_rot = x.copy()
    for i in range(0, d_model, 2):
        # 计算旋转角度
        theta = 1.0 / (10000 ** (2 * i / d_model))
        angle = pos * theta
        cos_t = np.cos(angle)
        sin_t = np.sin(angle)
        # 两两分组旋转
        x1 = x[i]
        x2 = x[i+1]
        x_rot[i] = x1 * cos_t - x2 * sin_t
        x_rot[i+1] = x1 * sin_t + x2 * cos_t
    return x_rot

# 对整个序列批量使用RoPE
def apply_rope_to_seq(seq: np.ndarray, d_model: int) -> np.ndarray:
    seq_len = seq.shape[0]
    rope_seq = np.zeros_like(seq)
    for pos in range(seq_len):
        rope_seq[pos] = rope_encoding(seq[pos], pos, d_model)
    return rope_seq

if __name__ == "__main__":
    # 定义参数：序列长度10，向量维度8

    max_seq_len = 10
    dim = 8

    # 调用函数生成正弦位置编码
    pe = sinusoidal_pos_encoding(max_seq_len, dim)
    print("===== 1. 正弦位置编码结果 =====")
    print(pe)

        # 测试二维向量旋转
    x1, x2 = 1.0, 0.0
    theta = np.pi / 4  # 旋转45度
    r1, r2 = rotate_2d(x1, x2, theta)
    print("\n===== 2. 二维向量旋转结果 =====")
    print(f"原始向量({x1}, {x2})，旋转后({r1:.4f}, {r2:.4f})")
    
        # 模拟词嵌入向量
    seq_embed = np.random.randn(max_seq_len, dim)
    # 应用RoPE
    rope_embed = apply_rope_to_seq(seq_embed, dim)
    print("\n===== 3. 高维RoPE编码结果 =====")
    print(rope_embed)

        # E + pos：词向量 + 正弦位置编码（加法融合）
    add_result = seq_embed + pe
    print("\n===== 4. E+pos 与 RoPE 方式对比 =====")
    print("E+pos：词向量 与 位置编码【直接相加】")
    print("RoPE：对原始向量【做旋转变换】，不使用加法")

        # 验证RoPE相对位置性质
    d = 8
    q_base = np.array([1.0] * d)
    k_base = np.array([1.0] * d)

    # 第一组：位置2 和 位置5，相对位置 = 3
    q2 = rope_encoding(q_base, pos=2, d_model=d)
    k5 = rope_encoding(k_base, pos=5, d_model=d)
    dot1 = np.dot(q2, k5)

    # 第二组：位置4 和 位置7，相对位置 = 3
    q4 = rope_encoding(q_base, pos=4, d_model=d)
    k7 = rope_encoding(k_base, pos=7, d_model=d)
    dot2 = np.dot(q4, k7)

    print("\n===== 5. RoPE 相对位置验证 =====")
    print(f"相对位置=3，(2,5)点积：{dot1:.4f}")
    print(f"相对位置=3，(4,7)点积：{dot2:.4f}")
    print("结论：相同相对位置，点积几乎相等，RoPE具备相对位置特性")