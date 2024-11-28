segs_path = ['path1', 'path2', 'path3']  # 示例原始列表
target_length = 30

# 填充或裁剪
padded_segs_path = segs_path + [''] * (target_length - len(segs_path))

print(padded_segs_path)
print(len(padded_segs_path))  # 应该输出 30
