import numpy as np
import matplotlib.pyplot as plt

#---------------------------
# 1. 数据准备与预处理
#---------------------------
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

patterns = []
for line in data_lines:
    arr = np.array([int(ch) for ch in line])
    arr = 2*arr - 1  # 0->-1, 1->+1
    patterns.append(arr)
patterns = np.array(patterns)

num_patterns = patterns.shape[0]
N = patterns.shape[1]

# 训练Hopfield网络
W = np.zeros((N, N))
for p in range(num_patterns):
    x = patterns[p].reshape(-1,1)
    W += x @ x.T
np.fill_diagonal(W, 0)

def update_state(x, W):
    new_x = np.sign(W @ x)
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

def add_noise(x, noise_ratio):
    x_noisy = x.copy()
    n_flip = int(len(x)*noise_ratio)
    flip_indices = np.random.choice(len(x), n_flip, replace=False)
    x_noisy[flip_indices] = -x_noisy[flip_indices]
    return x_noisy

def is_stored_pattern(x, patterns):
    return any(np.all(x == p) for p in patterns)

#---------------------------
# Monte Carlo 方法测试字符2和字符6的恢复概率
test_patterns = [2, 6]  # 测试的模式：2号和6号
noise_levels = np.arange(0.1, 0.501, 0.001)  # 从10%到50%，步长0.1%
num_trials = 100  # 每个噪声水平重复100次试验

results = {pattern: [] for pattern in test_patterns}

for pattern_index in test_patterns:
    for nl in noise_levels:
        successes = 0
        for _ in range(num_trials):
            x_init = add_noise(patterns[pattern_index], nl)
            x_out = run_hopfield(x_init, W)
            if np.all(x_out == patterns[pattern_index]):
                successes += 1
        success_rate = successes / num_trials
        results[pattern_index].append(success_rate)

#---------------------------
# 绘制结果图像
#---------------------------
plt.figure(figsize=(12, 6))
for pattern_index in test_patterns:
    plt.plot(noise_levels * 100, results[pattern_index], label=f'Pattern {pattern_index}', linewidth=2)

plt.title("Recovery Probability for Patterns 2 and 6 (Monte Carlo, 0.1% Step Size)", fontsize=14)
plt.xlabel("Noise Level (%)", fontsize=12)
plt.ylabel("Recovery Probability", fontsize=12)
plt.ylim(-0.05, 1.05)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()