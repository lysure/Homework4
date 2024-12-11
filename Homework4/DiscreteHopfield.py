import numpy as np

#---------------------------
# 1. 数据准备与预处理
#---------------------------

# 假设这是8个字符的二值编码数据，每个字符串为10×12=120位0/1的编码。
data_lines = [
"000000000000011110000011111100011100111001110011100111001110011100111001110011100111001110001111110000011110000000000000",
"000111100000011110000001111000000111100000011110000001111000000111100000011110000001111000000111100000011110000001111000",
"111111110011111111000000001100000000110000000011001111111100111111110011000000001100000000110000000011111111001111111100",
"001111110000111111100000000110000000011000000001100000111100000011110000000001100000000110000000011000111111100011111100",
"011000011001100001100110000110011000011001100001100111111110011111111000000001100000000110000000011000000001100000000110",
"111111000011111100001100000000110000000011000000001111110000111111000011001100001100110000110011000011111100001111110000",
"111110000011111000001111100000111110000011111000001111100000000000000000000000000000000000000000000000000000000000000000",
"000011111100001111110000110011000011001100001100110000111111000011111100000000110000000011000000001100001111110000111111"
]

# 将0/1转换为+1/-1
patterns = []
for line in data_lines:
    arr = np.array([int(ch) for ch in line])
    arr = 2*arr - 1  # 0->-1, 1->+1
    patterns.append(arr)
patterns = np.array(patterns)  # shape: (8, 120)

num_patterns = patterns.shape[0]
N = patterns.shape[1]  # N=120

#---------------------------
# 2. 利用外积法训练Hopfield网络
#---------------------------

W = np.zeros((N, N))
for p in range(num_patterns):
    x = patterns[p].reshape(-1,1)
    W += x @ x.T
# 清除对角线元素
np.fill_diagonal(W, 0)

#---------------------------
# 3. 定义状态更新与运行函数
#---------------------------

def update_state(x, W):
    # x: (N,) vector of +/-1
    new_x = np.sign(W @ x)
    # 若有0，将其置为+1
    new_x[new_x == 0] = 1
    return new_x

def run_hopfield(x_init, W, max_iter=100):
    x = x_init.copy()
    for _ in range(max_iter):
        new_x = update_state(x, W)
        if np.all(new_x == x):
            break
        x = new_x
    return x

#---------------------------
# 4. 无噪声测试
#---------------------------

print("Testing stored patterns without noise:")
for i in range(num_patterns):
    x_init = patterns[i]
    x_out = run_hopfield(x_init, W)
    is_recovered = np.all(x_out == x_init)
    print(f" Pattern {i} recovered from itself: {is_recovered}")

#---------------------------
# 5. 加噪声测试(10%, 20%)
#---------------------------

def add_noise(x, noise_ratio=0.1):
    x_noisy = x.copy()
    n_flip = int(len(x)*noise_ratio)
    flip_indices = np.random.choice(len(x), n_flip, replace=False)
    x_noisy[flip_indices] = -x_noisy[flip_indices]
    return x_noisy

def is_stored_pattern(x, patterns):
    return any(np.all(x == p) for p in patterns)

noise_levels = [0.1, 0.2]  # 10%和20%
for nl in noise_levels:
    print(f"\nTesting with noise level: {nl*100}%")
    for i in range(num_patterns):
        x_init = add_noise(patterns[i], noise_ratio=nl)
        x_out = run_hopfield(x_init, W)
        is_recovered = np.all(x_out == patterns[i])
        if not is_recovered:
            # 检查是否为伪吸引子
            if not is_stored_pattern(x_out, patterns):
                print(f" Pattern {i} ended in a spurious attractor at noise {nl*100}%.")
            else:
                print(f" Pattern {i} did not recover original but recovered another stored pattern?!")
        else:
            print(f" Pattern {i} recovered successfully at noise {nl*100}%")