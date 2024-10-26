import xml.etree.ElementTree as ET


def sum_masses_in_urdf(file_path):
    # 解析URDF文件
    tree = ET.parse(file_path)
    root = tree.getroot()

    # 初始化质量总和
    total_mass = 0.0

    # 遍历所有的'mass'标签并累计质量
    for mass in root.iter('mass'):
        mass_value = mass.get('value')
        if mass_value is not None:
            total_mass += float(mass_value)

    return total_mass

# 使用例子
file_path = "../../resources/robots/gr1t1/urdf/GR1T1_5DoF.urdf"
# file_path = "/home/mao/Github_Project/smooth-humanoid-locomotion/simulation/legged_gym/resources/robots/gr1t1/urdf/GR1T1_5dof.urdf"

print("Total mass:", sum_masses_in_urdf(file_path))
