import os

def generate_mesh_tags(folder_path):
    # 遍历指定文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 检查文件是否是以 .STL 结尾的文件（忽略大小写）
        if filename.lower().endswith(".stl"):
            # 去除文件扩展名，生成对应的 mesh name
            mesh_name = os.path.splitext(filename)[0]
            # 生成 <mesh> 标签
            mesh_tag = f'<mesh name="{mesh_name}" file="fourier_hand/{filename}"/>'
            # 输出生成的标签
            print(mesh_tag)

# 示例用法，指定文件夹路径
folder_path = "/home/mao/Github_Project/humanoid-gym/resources/robots/gr1t1/meshes/fourier_hand"  # 替换成你的STL文件夹路径
generate_mesh_tags(folder_path)

# import numpy as np
# rng_com = [-1,1]
# rand_com_x = np.random.uniform(rng_com[0], rng_com[1], size=(1, ))
# rand_com_y = np.random.uniform(rng_com[0], rng_com[1], size=(1, ))
# rand_com_z = np.random.uniform(rng_com[0], rng_com[1], size=(1, ))
# rand_com = np.concatenate((rand_com_x, rand_com_y, rand_com_z))
# print(rand_com)