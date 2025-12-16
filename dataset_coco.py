def merge_voc_ids():
    # 定义ImageSets/Main文件夹路径
    main_folder = '/home/hebut-02/common/gzj/datasets/VOCdevkit/VOC2012/ImageSets/Main/'
    # 定义要合并的文件名
    file_names = ['train.txt', 'val.txt', 'trainval.txt']
    all_ids = []
    # 遍历每个文件，读取其中的图像ID并添加到all_ids列表中
    for file_name in file_names:
        file_path = main_folder + file_name
        with open(file_path, 'r') as f:
            ids = [line.strip() for line in f.readlines()]
            all_ids.extend(ids)
    # 将所有图像ID去重后写入新的txt文件
    with open('all_voc2012_ids.txt', 'w') as f:
        for id in set(all_ids):
            f.write(id + '\n')


if __name__ == '__main__':
    # merge_voc_ids()
    # 创建一个空集合来存储合并后的内容（集合可以自动去重）
    merged_data = set()
    # 打开第一个文件并读取内容
    try:
        with open("all_voc2007_ids.txt", "r") as file1:
            for line in file1:
                merged_data.add(line.strip())
    except FileNotFoundError:
        print("all_voc2007_ids.txt文件不存在")
    # 打开第二个文件并读取内容
    try:
        with open("all_voc2012_ids.txt", "r") as file2:
            for line in file2:
                merged_data.add(line.strip())
    except FileNotFoundError:
        print("all_voc2012_ids.txt文件不存在")
    # 打开第三个文件并读取内容
    try:
        with open("instances_train2017.txt", "r") as file3:
            for line in file3:
                merged_data.add(line.strip())
    except FileNotFoundError:
        print("instances_train2017.txt文件不存在")
    # 将合并后的数据写入一个新的文件（可以自定义文件名）
    with open("merged_and_deduplicated.txt", "w") as output_file:
        for data in merged_data:
            output_file.write(data + "\n")