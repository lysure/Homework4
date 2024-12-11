import numpy as np

# --------------------------
# 数据准备
# --------------------------
# 假设您已从题中提取出8个字母的5x7编码（0/1），下面给出示例函数：
# 实际请用题中给出的字母点阵数据替换 X_data_dict 中对应字母的编码。
# X_data_dict中的value是长度为35的0/1列表，需要您从题给定的数据中填写。
X_data_dict = {
      'G': [0,1,1,1,0,
          1,0,0,0,1,
          1,0,0,0,1,
          1,1,1,1,1,
          1,0,0,0,1,
          1,0,0,0,1,
          1,0,0,0,1],
    'N': [0,1,1,1,0,
          1,0,0,0,1,
          1,0,0,0,1,
          1,0,1,0,1,
          1,0,0,1,1,
          1,0,0,0,1,
          1,0,0,0,1],
    'I': [0,1,1,1,0,
          0,1,0,0,1,
          0,1,0,0,1,
          0,1,1,1,0,
          0,1,0,0,1,
          0,1,0,0,1,
          0,1,1,1,0],
    'Q': [0,1,1,1,0,
          1,0,0,0,1,
          1,0,0,0,1,
          1,0,1,0,1,
          1,0,0,1,1,
          0,1,0,0,1,
          0,0,1,1,1],
    'O': [0,1,1,1,0,
          1,0,0,0,1,
          1,0,0,0,1,
          1,0,0,0,1,
          1,0,0,0,1,
          1,0,0,0,1,
          0,1,1,1,0],
    'U': [1,0,0,0,1,
          1,0,0,0,1,
          1,0,0,0,1,
          1,0,0,0,1,
          1,0,0,0,1,
          1,0,0,0,1,
          0,1,1,1,0],
    'H': [1,0,0,0,1,
          1,0,0,0,1,
          1,0,0,0,1,
          1,1,1,1,1,
          1,0,0,0,1,
          1,0,0,0,1,
          1,0,0,0,1],
    'Z': [1,1,1,1,1,
          0,0,0,1,0,
          0,0,1,0,0,
          0,1,0,0,0,
          1,0,0,0,0,
          1,0,0,0,0,
          1,1,1,1,1]
}

# 标号向量字典
Y_data_dict = {
    'G': [-1, -1, -1],
    'N': [-1, -1,  1],
    'I': [-1,  1, -1],
    'Q': [-1,  1,  1],
    'O': [ 1, -1, -1],
    'U': [ 1, -1,  1],
    'H': [ 1,  1, -1],
    'Z': [ 1,  1,  1]
}

# 将0/1转为±1
def bin_to_pm1(vec):
    return [1 if v == 1 else -1 for v in vec]

X_data = []
Y_data = []

letters = ['G','N','I','Q','O','U','H','Z'] # 使用的8个字母
for l in letters:
    x = bin_to_pm1(X_data_dict[l]) # 长度35
    y = Y_data_dict[l] # 长度3
    X_data.append(x)
    Y_data.append(y)

X_data = np.array(X_data) # (M,35)
Y_data = np.array(Y_data) # (M,3)

M = len(letters)

# --------------------------
# 构建BAM权值矩阵
# --------------------------
W = np.zeros((35,3)) # W为(35,3)
for i in range(M):
    x = X_data[i].reshape(35,1) # (35,1)
    y = Y_data[i].reshape(1,3)  # (1,3)
    W += x @ y  # 外积相加

# --------------------------
# 定义联想函数
# --------------------------
def sign_vec(v):
    return np.where(v>=0, 1, -1)

def forward_associate(X_input):
    # X_input: (35,) vector
    return sign_vec(W.T @ X_input)

def backward_associate(Y_input):
    # Y_input: (3,) vector
    return sign_vec(W @ Y_input)


# --------------------------
# 测试正向联想（无噪声）
# --------------------------
for i,l in enumerate(letters):
    X_in = X_data[i]
    Y_out = forward_associate(X_in)
    print(f"Letter {l}: Forward -> {Y_out}, Expected {Y_data[i]}")

# --------------------------
# 加噪声测试（正向）
# --------------------------
def add_noise_to_X(X_in, noise_ratio):
    # noise_ratio例如0.1或0.2,表示10%或20%的翻转
    X_noisy = X_in.copy()
    length = len(X_noisy)
    flip_count = int(length * noise_ratio)
    flip_idx = np.random.choice(length, flip_count, replace=False)
    for idx in flip_idx:
        X_noisy[idx] = -X_noisy[idx]
    return X_noisy

print("\n--- Noise Test (Forward) ---")
for noise_level in [0.1, 0.2]:
    print(f"Noise Level: {noise_level*100}%")
    for i,l in enumerate(letters):
        X_noisy = add_noise_to_X(X_data[i], noise_level)
        Y_out = forward_associate(X_noisy)
        correct = np.all(Y_out == Y_data[i])
        print(f" {l}: Correct={correct}, Output={Y_out}, Expected={Y_data[i]}")

# --------------------------
# 测试反向联想（从Y到X）
# --------------------------
print("\n--- Backward Association Test ---")
for i,l in enumerate(letters):
    Y_in = Y_data[i]
    X_out = backward_associate(Y_in)
    # 检查X_out与X_data[i]的匹配度
    match_count = np.sum(X_out == X_data[i])
    accuracy = match_count / 35.0
    print(f" {l}: Match {accuracy*100:.1f}%")

# 同样可对Y加入噪声进行测试
def add_noise_to_Y(Y_in, noise_count):
    Y_noisy = Y_in.copy()
    flip_idx = np.random.choice(len(Y_noisy), noise_count, replace=False)
    for idx in flip_idx:
        Y_noisy[idx] = -Y_noisy[idx]
    return Y_noisy

print("\n--- Noise Test (Backward) ---")
for noise_count in [1]: # 对3维向量翻转1位
    for i,l in enumerate(letters):
        Y_noisy = add_noise_to_Y(Y_data[i], noise_count)
        X_out = backward_associate(Y_noisy)
        match_count = np.sum(X_out == X_data[i])
        accuracy = match_count / 35.0
        print(f" {l} with {noise_count} noisy bit(s) in Y: Match {accuracy*100:.1f}%")