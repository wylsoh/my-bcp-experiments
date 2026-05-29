import os
import random

# 替换为你实际存放 Pancreas .h5 文件的目录
data_dir = "../data_split/Pancreas/Pancreas_h5"
output_dir = "../data_split/Pancreas/20percent"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 1. 获取所有病例名称 (去掉后缀)
all_cases = [f.split('.')[0] for f in os.listdir(data_dir) if f.endswith('.h5')]
all_cases.sort() # 先排序保证可复现性

# 2. 按照 62 训练，20 测试的常规比例切分 (这里为了打乱使用了特定的随机种子)
random.seed(1337)
random.shuffle(all_cases)

train_cases = all_cases[:62]
test_cases = all_cases[62:]

# 3. 划分 20% 有标签数据 (62 * 0.2 ≈ 12 个病例)
num_labeled = int(len(train_cases) * 0.2)
labeled_cases = train_cases[:num_labeled]
unlabeled_cases = train_cases[num_labeled:]

# 4. 写入文件函数
def write_txt(cases, filename):
    with open(os.path.join(output_dir, filename), 'w') as f:
        for case in cases:
            f.write(case + '\n')

write_txt(labeled_cases, 'train_lab.txt')
write_txt(unlabeled_cases, 'train_unlab.txt')
write_txt(test_cases, 'val.txt')

print(f"成功生成列表文件！\n有标签: {len(labeled_cases)} 个\n无标签: {len(unlabeled_cases)} 个\n测试集: {len(test_cases)} 个")