import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Songti SC']  # 使用黑体字体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
# Sigmoid 激活函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 前向传播函数，用于计算网络的输出
def forward_propagation(weights, inputs):
    # 提取权重参数
    w10, w12, w13, w20, w24, w25, w30, w34, w35 = weights
    # 计算隐藏层节点 2 和 3 的输出
    net2 = sigmoid(w20 * -1 + w24 * inputs[0] + w25 * inputs[1])
    net3 = sigmoid(w30 * -1 + w34 * inputs[0] + w35 * inputs[1])
    # 计算输出层节点 1 的输出
    net1 = sigmoid(w10 * -1 + w12 * net2 + w13 * net3)
    return net1

# 适应度函数（基于均方误差的倒数）
def fitness_function(weights, samples):
    total_error = 0
    # 遍历样本数据，计算总误差
    for inputs, expected_output in samples:
        predicted_output = forward_propagation(weights, inputs)
        total_error += (predicted_output - expected_output) ** 2
    # 返回适应度值（误差越小，适应度越高）
    return 1 / (1 + total_error)

# 初始化种群（随机生成权重）
def initialize_population(pop_size, num_weights):
    return np.random.uniform(-1, 1, (pop_size, num_weights))

# 选择函数（轮盘赌法选择适应度较高的个体）
def select(population, fitness_scores):
    probabilities = fitness_scores / np.sum(fitness_scores)
    indices = np.random.choice(len(population), size=len(population), p=probabilities)
    return population[indices]

# 交叉函数（单点交叉）
def crossover(parent1, parent2):
    point = np.random.randint(len(parent1))  # 随机选择交叉点
    child1 = np.concatenate((parent1[:point], parent2[point:]))
    child2 = np.concatenate((parent2[:point], parent1[point:]))
    return child1, child2

# 变异函数
def mutate(chromosome, mutation_rate=0.1):
    for i in range(len(chromosome)):
        if np.random.rand() < mutation_rate:  # 按概率进行变异
            chromosome[i] += np.random.uniform(-0.5, 0.5)  # 随机调整权重
    return chromosome

# 遗传算法主函数
def genetic_algorithm(samples, num_generations=1000, pop_size=100, mutation_rate=0.1, crossover_rate=0.8):
    num_weights = 9  # 神经网络的权重个数
    population = initialize_population(pop_size, num_weights)  # 初始化种群
    best_fitness_per_generation = []  # 用于记录每代的最佳适应度

    for generation in range(num_generations):
        fitness_scores = np.array([fitness_function(ind, samples) for ind in population])

        # 记录当前代的最佳适应度
        best_fitness_per_generation.append(np.max(fitness_scores))

        # 选择
        population = select(population, fitness_scores)

        # 交叉
        next_generation = []
        for i in range(0, len(population), 2):
            if np.random.rand() < crossover_rate:  # 按交叉概率进行交叉
                parent1, parent2 = population[i], population[(i + 1) % len(population)]
                child1, child2 = crossover(parent1, parent2)
                next_generation.extend([child1, child2])
            else:
                next_generation.extend([population[i], population[(i + 1) % len(population)]])

        # 变异
        population = np.array([mutate(ind, mutation_rate) for ind in next_generation])

    # 返回最佳权重和每代的最佳适应度变化
    best_index = np.argmax(fitness_scores)
    return population[best_index], best_fitness_per_generation

# 异或问题数据集
samples = [
    ([1, 1], 0),
    ([-1, 1], 1),
    ([1, -1], 1),
    ([-1, -1], 0),
]

# 参数组合进行实验
results = {}

parameters = [
    {"pop_size": 50, "mutation_rate": 0.05, "crossover_rate": 0.8},
    {"pop_size": 100, "mutation_rate": 0.1, "crossover_rate": 0.8},
    {"pop_size": 200, "mutation_rate": 0.2, "crossover_rate": 0.8},
    {"pop_size": 100, "mutation_rate": 0.1, "crossover_rate": 0.6},
]

# 遍历不同参数组合，运行遗传算法
for param in parameters:
    best_weights, fitness_progression = genetic_algorithm(
        samples,
        num_generations=500,
        pop_size=param["pop_size"],
        mutation_rate=param["mutation_rate"],
        crossover_rate=param["crossover_rate"],
    )
    results[str(param)] = fitness_progression

# 绘制结果图
plt.figure(figsize=(10, 6))
for param, progression in results.items():
    plt.plot(progression, label=f"{param}")
plt.xlabel("Generation (代数)")
plt.ylabel("Best Fitness (最佳适应度)")
plt.title("Genetic Algorithm Performance with Different Parameters (遗传算法性能)")
plt.legend()
plt.show()