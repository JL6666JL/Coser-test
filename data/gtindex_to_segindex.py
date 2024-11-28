index_convert = [0]
seg_info = "/home/jianglei/work/CoSeR-test/data/ImageNet_10/Obj512_all/seg_all.txt"
convert_info = "/home/jianglei/work/CoSeR-test/data/ImageNet_10/Obj512_all/convert.txt"
prefix = ""

with open(seg_info,"r") as fin:
    for index, line in enumerate(fin.readlines()):
        if index == 0:
            underline = line.find("_", line.find("_") + 1)
            prefix = line[:underline]
        else:
            if prefix == "":
                print('something wrong!')
                break
            else:
                underline = line.find("_", line.find("_") + 1)
                now_prefix = line[:underline]
                if now_prefix != prefix:
                    prefix = now_prefix
                    index_convert.append(index)
                    
with open(seg_info,"r") as fin:
    index_convert.append(len(fin.readlines()))

with open(convert_info,"w") as fin:
    for index in index_convert:
        fin.write(f"{index}\n")



