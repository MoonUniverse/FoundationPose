def scale_obj(input_filename, output_filename, scale_factor):
    with open(input_filename, 'r') as file:
        lines = file.readlines()

    with open(output_filename, 'w') as file:
        for line in lines:
            if line.startswith('v '):  # 顶点数据行
                parts = line.split()
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                # 缩放顶点坐标
                x, y, z = x / scale_factor, y / scale_factor, z / scale_factor
                # 写入新的顶点数据
                file.write(f'v {x} {y} {z}\n')
            else:
                # 直接写入其他行
                file.write(line)

# 使用示例
input_filename = '/home/ubuntu/Documents/github/FoundationPose/demo_data/station/mesh/3dpea.obj'  # 你的原OBJ文件名
output_filename = '/home/ubuntu/Documents/github/FoundationPose/demo_data/station/mesh/3dpea_scaled_model.obj'  # 输出的OBJ文件名
scale_factor = 1000  # 从mm到m的缩放因子
scale_obj(input_filename, output_filename, scale_factor)