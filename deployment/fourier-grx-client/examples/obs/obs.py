import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib
# 使图片单独在窗口中显示
# matplotlib.use('TkAgg')
import csv


def read_csv(file_path):
    data = []

    # 打开csv文件并读取所有行数据
    with open(file_path, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            # 将每一行的数据转换为浮点数并加入到data列表
            data.append([float(x) for x in row])

    return data


def plot_group(data, start, end, title, xlabel='Time Step', ylabel='Value'):
    # 将数据在指定范围内进行绘制
    data_group = [row[start:end] for row in data]
    plt.figure(figsize=(10, 6))

    # 遍历每一列进行绘制
    for i in range(len(data_group[0])):
        plt.plot([row[i] for row in data_group], label=f'Column {start + i}')

    # 设置标题和标签
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # 显示图例
    plt.legend()

    # 显示图形
    # plt.show()
    plt.savefig('pic/' + title+'png')

def plot_all_groups(data):
    # 分别绘制每一组数据
    plot_group(data, 0, 2, 'Right Leg Phase and Left Leg Phase (0-1)')
    plot_group(data, 2, 5, 'Command Velocities (2-4)')
    plot_group(data, 5, 8, 'Angular Velocities (5-8)')
    plot_group(data, 8, 10, 'Euler Angles (8-10)')
    plot_group(data, 10, 16, 'Left Joint Positions (10-16)')
    plot_group(data, 16, 22, 'Right Joint Positions (16-22)')
    plot_group(data, 22, 28, 'Left Joint Velocities (22-28)')
    plot_group(data, 28, 34, 'Right Joint Velocities (28-34)')
    plot_group(data, 34, 39, 'Left Actions (34-39)')
    plot_group(data, 39, 44, 'Right Actions (39-44)')


    # plt.show()

def plot_target_pos(obs_file):
    # 读取CSV文件
    obs = pd.read_csv(obs_file, header=None)
    joint_default_position = np.array([0.0, 0.0, -0.4, 0.8, -0.4, -0.0,  # left leg (6)
                                               -0.0, 0.0, -0.4, 0.8, -0.4, 0.0,])  # right leg (6)
    obs.iloc[:,10:22] += joint_default_position
    # plt.style.use('default')
    # 创建每个关节的对比图
    for i in range(12):
        plt.figure(figsize=(8, 4))  # 设置图形大小
        plt.gca().set_facecolor('white')
        plt.plot(obs.iloc[:200, 44+i], marker='o',color='blue', label='Target Pos')
        plt.plot(obs.iloc[:200, 10+i], marker='o', color='red', label='Current Pos')
        plt.title(f'Pos {i + 1} Comparison')
        plt.xlabel('Sample Index')
        plt.ylabel('Pos Value')
        # plt.xticks(range(len(ideal_tau)))  # 根据数据量调整
        plt.legend()
        plt.grid()
        # plt.show()  # 显示每个图
        plt.savefig('pos/' + str(i) +'png')


def plot_joint_comparison(ideal_file, current_file):
    # 读取CSV文件
    ideal_tau = pd.read_csv(ideal_file, header=None)
    cur_tau = pd.read_csv(current_file, header=None)

    # 确保数据行数相同
    if ideal_tau.shape[0] != cur_tau.shape[0]:
        raise ValueError("两个文件的行数不匹配！")

    num_joints = ideal_tau.shape[1]
    # plt.style.use('default')
    # 创建每个关节的对比图
    for i in range(num_joints):
        plt.figure(figsize=(8, 4))  # 设置图形大小
        plt.gca().set_facecolor('white')
        plt.plot(ideal_tau.iloc[:, i], color='blue', label='Ideal Tau')
        plt.plot(cur_tau.iloc[:, i], color='red', label='Current Tau')
        plt.title(f'Joint {i + 1} Comparison')
        plt.xlabel('Sample Index')
        plt.ylabel('Tau Value')
        # plt.xticks(range(len(ideal_tau)))  # 根据数据量调整
        plt.legend()
        plt.grid()
        # plt.show()  # 显示每个图
        plt.savefig('tau/' + str(i) +'png')

# 文件路径
csv_file_path = 'obs.csv'

# 从csv中读取数据
obs_data = read_csv(csv_file_path)
# plot_target_pos(csv_file_path)

# 绘制所有分组数据
plot_all_groups(obs_data)

ideal_file = 'ideal_tau.csv'
current_file = 'cur_tau.csv'
plot_joint_comparison(ideal_file, current_file)
