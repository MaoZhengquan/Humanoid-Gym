import matplotlib.pyplot as plt
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
    plt.show()
    # plt.savefig(title+'png')

def plot_all_groups(data):
    # 分别绘制每一组数据
    plot_group(data, 0, 2, 'Right Leg Phase and Left Leg Phase (0-1)')
    plot_group(data, 2, 5, 'Command Velocities (2-4)')
    plot_group(data, 5, 11, 'Left Joint Positions (5-11)')
    plot_group(data, 11, 17, 'Right Joint Positions (11-17)')

    plot_group(data, 17, 23, 'Left Joint Velocities (17-23)')
    plot_group(data, 23, 29, 'Right Joint Velocities (23-29)')

    plot_group(data, 29, 32, 'Angular Velocities (29-32)')
    plot_group(data, 32, 35, 'Euler Angles (32-35)')
    # plot_group(data, 35, 47, 'Actions (35-47)')

    plot_group(data, 35, 41, 'Actions (35-39)')
    plot_group(data, 41, 47, 'Actions (41-45)')
    # plt.show()

# 文件路径
csv_file_path = 'obs.csv'

# 从csv中读取数据
obs_data = read_csv(csv_file_path)

# 绘制所有分组数据
plot_all_groups(obs_data)