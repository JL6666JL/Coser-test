# train_info = "/home/jianglei/work/CoSeR/data/ImageNet/Obj512_all/train.txt"
# seg_info = "/home/jianglei/work/CoSeR/data/ImageNet/Obj512_all/seg_all.txt"
# train_seg_info = "/home/jianglei/work/CoSeR/data/ImageNet/Obj512_all/train_seg.txt"
train_info = "/home/jianglei/work/CoSeR-test/data/ImageNet_10/Obj512_all/train.txt"
seg_info = "/home/jianglei/work/CoSeR-test/data/ImageNet_10/Obj512_all/seg_all.txt"
train_seg_info = "/home/jianglei/work/CoSeR-test/data/ImageNet_10/Obj512_all/train_seg.txt"
with open(train_info,'r') as f:
    train_files = [line.strip() for line in f.readlines()]

with open(seg_info,'r') as f:
    seg_files = [line.strip() for line in f.readlines()]


seg_dict = {}
for seg_file in seg_files:
    prefix = "_".join(seg_file.split("_")[:2])
    if prefix not in seg_dict:
        seg_dict[prefix] = []
    seg_dict[prefix].append(seg_file)

# 匹配并存储结果到 train_seg.txt
with open(train_seg_info, "w") as f:
    for train_file in train_files:
        if '/' in train_file:
            print("no")
        prefix = "_".join(train_file.split("_")[:2]).replace(".JPEG","")
        if prefix in seg_dict:
            segs=""
            for seg_file in seg_dict[prefix]:
                if '/' in seg_file:
                    print('no')
                segs =segs + '/'+ seg_file
            f.write(f"{train_file}{segs}\n")