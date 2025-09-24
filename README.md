#YOLO数据增强使用指南
##下面我将详细介绍如何使用上面的代码进行YOLO数据增强。

##1. 准备工作
###1.1 安装必要的库
bash
pip install albumentations opencv-python Pillow numpy
###1.2 准备数据集
确保您的YOLO数据集目录结构如下：

text
your_dataset/
├── images/          # 存放图像文件
│   ├── image1.jpg
│   ├── image2.jpg
│   ├── image3.jpg
│   └── ...
└── labels/          # 存放标注文件
    ├── image1.txt
    ├── image2.txt
    ├── image3.txt
    └── ...
###1.3 创建Python脚本
将上面的完整代码保存为yolo_augmentation.py

##2. 基本使用方法
###2.1 简单增强（推荐新手）
创建一个简单的使用脚本simple_augment.py：

python
from yolo_augmentation import YOLODatasetAugmentor

def main():
    # 创建增强器（使用默认参数）
    augmentor = YOLODatasetAugmentor(
        image_size=640,                    # 输出图像尺寸
        augmentations_per_image=3,         # 每张原始图像生成3张增强图像
        use_mosaic=True,                   # 启用Mosaic增强
        use_mixup=True,                    # 启用MixUp增强
        mosaic_prob=0.2,                   # 20%的概率使用Mosaic增强
        mixup_prob=0.2                     # 20%的概率使用MixUp增强
    )
    
    # 增强数据集
    augmentor.augment_dataset(
        dataset_dir='path/to/your/dataset',  # 替换为您的数据集路径
        output_dir='path/to/augmented_dataset',  # 替换为输出路径
        copy_original=True  # 将原始数据也复制到输出目录
    )
    
    print("数据增强完成！")

if __name__ == "__main__":
    main()
###2.2 运行脚本
bash
python simple_augment.py
##3. 高级使用方法
###3.1 自定义增强参数
创建custom_augment.py：

python
from yolo_augmentation import YOLODatasetAugmentor

def custom_augmentation():
    # 创建自定义增强器
    augmentor = YOLODatasetAugmentor(
        image_size=640,                    # 输出图像尺寸
        augmentations_per_image=5,         # 每张原始图像生成5张增强图像
        use_mosaic=True,                   # 启用Mosaic增强
        use_mixup=True,                    # 启用MixUp增强
        mosaic_prob=0.3,                   # 30%的概率使用Mosaic增强
        mixup_prob=0.2                     # 20%的概率使用MixUp增强
    )
    
    # 增强数据集
    augmentor.augment_dataset(
        dataset_dir='./my_yolo_dataset',
        output_dir='./augmented_dataset',
        copy_original=True,
        image_extensions=['.jpg', '.jpeg', '.png']  # 指定图像格式
    )

def test_single_image():
    """测试单张图像增强效果"""
    from yolo_augmentation import YOLOAugmentor
    import matplotlib.pyplot as plt
    
    augmentor = YOLOAugmentor(image_size=640)
    
    # 替换为您的测试图像和标注路径
    image_path = './my_yolo_dataset/images/image1.jpg'
    annotation_path = './my_yolo_dataset/labels/image1.txt'
    
    # 应用增强
    augmented_image, bboxes, labels = augmentor.augment_single_image(
        image_path, annotation_path,
        use_geometric=True,    # 使用几何变换
        use_color=True,        # 使用颜色变换
        use_noise=True,        # 使用噪声
        use_weather=False      # 不使用天气效果（可选）
    )
    
    # 显示结果对比
    import cv2
    original_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(original_image)
    plt.title('原始图像')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(augmented_image)
    plt.title('增强后图像')
    plt.axis('off')
    
    # 显示边界框（如果有）
    if bboxes:
        from matplotlib.patches import Rectangle
        plt.subplot(1, 3, 3)
        plt.imshow(augmented_image)
        ax = plt.gca()
        for bbox in bboxes:
            x_center, y_center, width, height = bbox
            x = (x_center - width/2) * 640
            y = (y_center - height/2) * 640
            w = width * 640
            h = height * 640
            rect = Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
        plt.title('增强后图像（带边界框）')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 运行自定义增强
    custom_augmentation()
    
    # 测试单张图像增强效果（可选）
    # test_single_image()
###3.2 分步增强（更精细的控制）
创建step_by_step_augment.py：

python
import cv2
import os
from pathlib import Path
from yolo_augmentation import YOLOAugmentor, MosaicAugmentor, MixUpAugmentor

def step_by_step_augmentation():
    """分步进行数据增强，提供更精细的控制"""
    
    # 数据集路径
    dataset_dir = './my_yolo_dataset'
    output_dir = './step_augmented_dataset'
    
    # 创建输出目录
    os.makedirs(f'{output_dir}/images', exist_ok=True)
    os.makedirs(f'{output_dir}/labels', exist_ok=True)
    
    # 初始化增强器
    single_augmentor = YOLOAugmentor(image_size=640)
    mosaic_augmentor = MosaicAugmentor(image_size=640)
    mixup_augmentor = MixUpAugmentor(image_size=640)
    
    # 获取所有图像文件
    image_files = list(Path(f'{dataset_dir}/images').glob('*.jpg'))
    image_files.extend(Path(f'{dataset_dir}/images').glob('*.png'))
    image_files.extend(Path(f'{dataset_dir}/images').glob('*.jpeg'))
    
    print(f"找到 {len(image_files)} 张图像")
    
    # 复制原始数据
    print("复制原始数据...")
    for img_path in image_files:
        # 复制图像
        ann_path = Path(f'{dataset_dir}/labels/{img_path.stem}.txt')
        if ann_path.exists():
            # 复制到输出目录
            cv2.imwrite(f'{output_dir}/images/{img_path.name}', cv2.imread(str(img_path)))
            single_augmentor.save_yolo_annotation(
                *single_augmentor.parse_yolo_annotation(str(ann_path)),
                f'{output_dir}/labels/{img_path.stem}.txt'
            )
    
    # 单图增强
    print("进行单图增强...")
    augmentation_count = 0
    for img_path in image_files:
        ann_path = Path(f'{dataset_dir}/labels/{img_path.stem}.txt')
        
        if not ann_path.exists():
            continue
            
        # 为每张图像生成2个增强版本
        for i in range(2):
            try:
                augmented_image, bboxes, labels = single_augmentor.augment_single_image(
                    str(img_path), str(ann_path)
                )
                
                # 保存增强结果
                aug_name = f"{img_path.stem}_single_aug_{i}{img_path.suffix}"
                cv2.imwrite(f'{output_dir}/images/{aug_name}', 
                           cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR))
                single_augmentor.save_yolo_annotation(
                    bboxes, labels, f'{output_dir}/labels/{img_path.stem}_single_aug_{i}.txt'
                )
                
                augmentation_count += 1
            except Exception as e:
                print(f"单图增强失败 {img_path}: {e}")
    
    # Mosaic增强（需要至少4张图像）
    print("进行Mosaic增强...")
    if len(image_files) >= 4:
        mosaic_count = min(10, len(image_files) // 4)  # 最多生成10个mosaic
        for i in range(mosaic_count):
            try:
                # 随机选择4张图像
                selected_images = []
                selected_annotations = []
                
                for j in range(4):
                    img_idx = (i * 4 + j) % len(image_files)
                    img_path = image_files[img_idx]
                    ann_path = Path(f'{dataset_dir}/labels/{img_path.stem}.txt')
                    
                    if ann_path.exists():
                        selected_images.append(str(img_path))
                        selected_annotations.append(str(ann_path))
                
                if len(selected_images) == 4:
                    mosaic_image, bboxes, labels = mosaic_augmentor.create_mosaic(
                        selected_images, selected_annotations
                    )
                    
                    # 保存mosaic结果
                    mosaic_name = f"mosaic_{i}.jpg"
                    cv2.imwrite(f'{output_dir}/images/{mosaic_name}', 
                               cv2.cvtColor(mosaic_image, cv2.COLOR_RGB2BGR))
                    single_augmentor.save_yolo_annotation(
                        bboxes, labels, f'{output_dir}/labels/mosaic_{i}.txt'
                    )
                    
                    augmentation_count += 1
            except Exception as e:
                print(f"Mosaic增强失败: {e}")
    
    print(f"增强完成！共生成 {augmentation_count} 张增强图像")

if __name__ == "__main__":
    step_by_step_augmentation()
##4. 验证增强结果
创建verify_augmentation.py来验证增强结果：

python
import cv2
import os
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def verify_augmentation(dataset_dir):
    """验证增强结果是否正确"""
    
    images_dir = Path(dataset_dir) / 'images'
    labels_dir = Path(dataset_dir) / 'labels'
    
    # 获取前5张增强图像进行验证
    image_files = list(images_dir.glob('*aug*.jpg'))[:5]
    image_files.extend(list(images_dir.glob('*mosaic*.jpg'))[:2])
    image_files.extend(list(images_dir.glob('*mixup*.jpg'))[:2])
    
    if not image_files:
        print("未找到增强图像，请检查输出目录")
        return
    
    # 创建验证图
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.ravel()
    
    for i, img_path in enumerate(image_files[:9]):  # 最多显示9张
        if i >= 9:
            break
            
        # 读取图像
        image = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        
        # 查找对应的标注文件
        label_path = labels_dir / f"{img_path.stem}.txt"
        
        # 显示图像
        axes[i].imshow(image)
        axes[i].set_title(f'{img_path.name}')
        axes[i].axis('off')
        
        # 显示边界框
        if label_path.exists():
            with open(label_path, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                data = line.strip().split()
                if len(data) == 5:
                    class_id, x_center, y_center, width, height = map(float, data)
                    
                    # 转换为像素坐标
                    img_height, img_width = image.shape[:2]
                    x = (x_center - width/2) * img_width
                    y = (y_center - height/2) * img_height
                    w = width * img_width
                    h = height * img_height
                    
                    # 添加边界框
                    rect = Rectangle((x, y), w, h, linewidth=2, 
                                   edgecolor='red', facecolor='none')
                    axes[i].add_patch(rect)
                    
                    # 添加类别标签
                    axes[i].text(x, y-5, f'Class {int(class_id)}', 
                               color='red', fontsize=10, weight='bold')
    
    # 隐藏多余的子图
    for i in range(len(image_files), 9):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('augmentation_verification.jpg', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"验证完成！结果已保存为 'augmentation_verification.jpg'")
    print(f"共增强图像数量: {len(list(images_dir.glob(\"*.jpg\")))}")
    print(f"共增强标注数量: {len(list(labels_dir.glob(\"*.txt\")))}")

if __name__ == "__main__":
    verify_augmentation('./augmented_dataset')  # 替换为您的增强数据集路径
##5. 实际应用示例
5.1 针对小数据集的增强策略
python
from yolo_augmentation import YOLODatasetAugmentor

def augment_small_dataset():
    """针对小数据集（<100张图像）的增强策略"""
    
    augmentor = YOLODatasetAugmentor(
        image_size=640,
        augmentations_per_image=10,  # 生成更多增强图像
        use_mosaic=True,
        use_mixup=True,
        mosaic_prob=0.4,  # 提高Mosaic概率
        mixup_prob=0.3    # 提高MixUp概率
    )
    
    augmentor.augment_dataset(
        dataset_dir='./small_dataset',
        output_dir='./augmented_small_dataset',
        copy_original=True
    )

def augment_large_dataset():
    """针对大数据集（>1000张图像）的增强策略"""
    
    augmentor = YOLODatasetAugmentor(
        image_size=640,
        augmentations_per_image=2,  # 生成较少增强图像
        use_mosaic=True,
        use_mixup=True,
        mosaic_prob=0.1,  # 降低Mosaic概率
        mixup_prob=0.1    # 降低MixUp概率
    )
    
    augmentor.augment_dataset(
        dataset_dir='./large_dataset',
        output_dir='./augmented_large_dataset',
        copy_original=False  # 不复制原始数据
    )
##6. 使用提示和注意事项
内存管理：如果数据集很大，可以考虑分批处理

增强强度：根据数据集大小调整增强强度

验证结果：务必使用验证脚本检查增强结果

备份数据：增强前备份原始数据集

文件命名：确保图像和标注文件一一对应

通过以上步骤，您可以轻松地使用这个YOLO数据增强工具来增强您的数据集，提高模型的泛化能力。

我使用的这个3.2 分步增强（更精细的控制），使用了验证的代码，帮我写成一个readme
我们已将3.2分步增强和验证代码整合，并需要编写一个README文件来说明如何使用。
注意：3.2分步增强代码是“step_by_step_augment.py”，验证代码是“verify_augmentation.py”。
我们将创建一个README.md文件，详细说明如何运行代码以及目录结构要求。

YOLO数据增强工具使用说明
项目概述
这是一个基于albumentations的YOLO格式数据集增强工具，支持单图增强、Mosaic拼接和MixUp混合等多种数据增强方式，能够显著提高目标检测模型的泛化能力。

功能特点
✅ 支持YOLO格式数据集

✅ 单图增强（几何变换、颜色变换、噪声添加等）

✅ Mosaic增强（4图拼接）

✅ MixUp增强（2图混合）

✅ 自动处理标注文件

✅ 可视化验证增强效果

✅ 灵活的参数配置

环境要求
Python 3.6+

albumentations

opencv-python

numpy

matplotlib

安装依赖
bash
pip install albumentations opencv-python numpy matplotlib
项目结构
text
yolo_augmentation/
├── yolo_augmentation.py    # 主要增强代码
├── step_by_step_augment.py # 分步增强脚本
├── verify_augmentation.py  # 验证脚本
└── README.md              # 说明文档
数据集准备
确保您的YOLO数据集目录结构如下：

text
your_dataset/
├── images/          # 存放图像文件
│   ├── image1.jpg
│   ├── image2.jpg
│   ├── image3.jpg
│   └── ...
└── labels/          # 存放标注文件
    ├── image1.txt
    ├── image2.txt
    ├── image3.txt
    └── ...
快速开始
方法一：分步增强（推荐）
运行分步增强脚本

bash
python step_by_step_augment.py
这个脚本会：

自动检测数据集目录结构

复制原始数据到输出目录

进行单图增强（每张图生成2个增强版本）

进行Mosaic增强（最多生成10个mosaic）

保存增强后的图像和标注文件

修改配置（可选）

如果需要自定义参数，编辑step_by_step_augment.py中的以下部分：

python
# 数据集路径
dataset_dir = './my_yolo_dataset'      # 修改为您的数据集路径
output_dir = './step_augmented_dataset' # 修改为输出路径

# 增强参数
image_size = 640                       # 输出图像尺寸
single_aug_count = 2                   # 单图增强数量
mosaic_count = 10                      # Mosaic增强数量
方法二：使用完整增强器
如果您需要更复杂的增强策略，可以使用完整增强器：

python
from yolo_augmentation import YOLODatasetAugmentor

# 创建增强器
augmentor = YOLODatasetAugmentor(
    image_size=640,
    augmentations_per_image=5,  # 每张原始图像生成5张增强图像
    use_mosaic=True,           # 启用Mosaic增强
    use_mixup=True,            # 启用MixUp增强
    mosaic_prob=0.3,           # Mosaic增强概率
    mixup_prob=0.2             # MixUp增强概率
)

# 增强数据集
augmentor.augment_dataset(
    dataset_dir='./my_yolo_dataset',
    output_dir='./augmented_dataset',
    copy_original=True  # 是否复制原始数据
)
验证增强结果
增强完成后，使用验证脚本检查增强效果：

bash
python verify_augmentation.py
验证脚本会：

随机选择增强后的图像和标注

显示图像和边界框

生成验证报告图像

统计增强数量

输出结果
增强后的数据集目录结构：

text
augmented_dataset/
├── images/
│   ├── image1.jpg              # 原始图像（如果选择复制）
│   ├── image1_single_aug_0.jpg # 单图增强结果
│   ├── image1_single_aug_1.jpg
│   ├── mosaic_0.jpg           # Mosaic增强结果
│   └── ...
└── labels/
    ├── image1.txt
    ├── image1_single_aug_0.txt
    ├── image1_single_aug_1.txt
    ├── mosaic_0.txt
    └── ...
增强类型说明
1. 单图增强
几何变换：翻转、旋转、缩放、剪切等

颜色变换：亮度、对比度、饱和度、色调调整

噪声添加：高斯噪声、运动模糊等

天气效果：雨、雾、眩光等（可选）

2. Mosaic增强
将4张图像拼接成一张2x2的网格，模拟复杂场景，提高模型对多目标和小目标的检测能力。

3. MixUp增强
将两张图像按随机比例混合，增加数据多样性，提高模型鲁棒性。

参数调优建议
小数据集（<100张图像）
python
augmentations_per_image = 10    # 生成更多增强图像
mosaic_prob = 0.4              # 提高Mosaic概率
mixup_prob = 0.3               # 提高MixUp概率
大数据集（>1000张图像）
python
augmentations_per_image = 2     # 生成较少增强图像
mosaic_prob = 0.1              # 降低Mosaic概率
mixup_prob = 0.1               # 降低MixUp概率
特定场景调整
室内场景：减少天气效果，增加颜色变换

室外场景：增加天气效果，适度使用几何变换

小目标检测：增加Mosaic增强概率

常见问题
Q1: 增强后边界框位置不正确
A: 检查原始标注文件格式是否正确，确保坐标值在0-1范围内。

Q2: 内存不足错误
A: 减少augmentations_per_image参数值，或分批处理大数据集。

Q3: 增强效果不理想
A: 调整增强参数，减少过于激进的变换，或使用验证脚本检查具体问题。

Q4: 找不到标注文件
A: 确保图像和标注文件一一对应，且位于正确的目录结构中。

高级用法
自定义增强管道
您可以修改yolo_augmentation.py中的增强管道：

python
# 更强的几何变换
self.geometric_transform = A.Compose([
    A.HorizontalFlip(p=0.7),
    A.ShiftScaleRotate(
        shift_limit=0.2,    # 增加平移幅度
        scale_limit=0.3,    # 增加缩放幅度
        rotate_limit=30,    # 增加旋转角度
        p=0.7              # 提高概率
    ),
    # ... 其他变换
])
添加新的增强方式
在YOLOAugmentor类中添加新的变换：

python
# 添加CutMix增强
self.cutmix_transform = A.Compose([
    A.Cutout(num_holes=8, max_h_size=32, max_w_size=32, p=0.5),
])
技术支持
如遇到问题，请检查：

数据集目录结构是否正确

依赖库版本是否兼容

图像和标注文件是否一一对应

标注文件格式是否符合YOLO标准

许可证
本项目仅供学习和研究使用。

图片数据增强
羊毛图像数据增强工具 README
项目介绍
本工具是一款专门针对羊毛图像的数据增强解决方案，旨在解决羊毛图像数据集数量不足、场景多样性欠缺的问题，提升后续模型训练（如羊毛质量检测、纤维分类等任务）的鲁棒性。工具内置多种适配羊毛特性的增强策略，支持轻 / 中 / 重度增强、纤维结构保护、环境模拟（雨雪 / 光照变化）等功能，并提供二次增强流程进一步扩展数据多样性。
核心功能
针对羊毛图像的特殊性（如纤维结构、颜色特征、质量指标关联性），工具提供以下核心增强能力：
增强策略	适用场景	核心特点
基础强度增强	通用羊毛图像数据扩充	分轻 / 中 / 重三档，平衡多样性与特征保留
纤维感知增强	细羊毛（纤维结构敏感）	限制过度几何变换，保护纤维方向与纹理细节
质量检测增强	羊毛质量分级、缺陷检测任务	强化纹理 / 颜色对比度，突出质量相关特征
环境模拟增强	模拟真实拍摄环境变化	支持雨天、雪天、光照波动等环境效果模拟
二次增强流程	需进一步提升数据多样性的场景	基于一次增强结果叠加环境模拟，扩大数据分布
环境依赖
1. Python 版本
推荐 Python 3.6 ~ 3.10（兼容主流数据科学库版本）。
2. 依赖库安装
使用 pip 安装以下依赖（建议添加国内镜像源加速，如清华源）：
bash
# 基础依赖安装（含国内镜像源，避免SSL问题）
pip install albumentations opencv-python numpy pathlib -i https://pypi.tuna.tsinghua.edu.cn/simple/
albumentations：核心图像增强库，提供高效的几何 / 颜色变换接口
opencv-python：图像读取、格式转换与保存
numpy：图像数组处理
pathlib：跨平台路径管理
快速开始
1. 目录准备
准备输入文件夹：存放原始羊毛图像（支持格式：jpg/jpeg/png/bmp/tiff）。
规划输出文件夹：用于保存增强后的图像（工具会自动创建，无需手动新建）。
⚠️ 注意：避免使用中文路径（如 “数据增强”），可能导致图像读取失败，建议使用英文路径（如 D:/Galaxy/wool_data/input）。
2. 基础使用示例
直接运行脚本，修改 if __name__ == "__main__": 下的参数即可快速执行增强：
python
if __name__ == "__main__":
    # 1. 第一次增强（基础强度增强，以重度为例）
    process_wool_images(
        input_folder="D:/Galaxy/wool_data/original",  # 原始图像路径
        output_folder="D:/Galaxy/wool_data/aug_first",# 一次增强输出路径
        augmentation_strength="heavy",                # 增强强度：light/moderate/heavy
        target_multiplier=10,                         # 目标总增强图像数（按原始图像均分）
        wool_type="coarse"                            # 羊毛类型：fine（细羊毛）/coarse（粗羊毛）/None（通用）
    )

    # 2. 第二次增强（环境模拟增强，基于一次增强结果）
    print("第一次增强完成，开始二次环境增强...")
    second_environment_process(
        input_folder="D:/Galaxy/wool_data/aug_first", # 一次增强结果路径（作为二次输入）
        output_folder="D:/Galaxy/wool_data/aug_second",# 二次增强输出路径
        target_multiplier=50                          # 二次增强目标总图像数
    )
3. 运行脚本
在终端进入脚本所在目录，执行以下命令：
bash
python step_by_step_augment.py
运行过程中会打印日志，示例如下：
plaintext
找到 20 张羊毛图像
增强强度: heavy
目标增强倍数: 10
增强完成! 共生成 200 张增强图像
原始图像数量: 20
平均每张原始图像生成: 10.0 张增强图像
第一次增强完成，开始第二次环境增强...
找到 200 张羊毛图像
增强强度: environment
目标增强倍数: 50
增强完成! 共生成 500 张增强图像
原始图像数量: 200
平均每张原始图像生成: 2.5 张增强图像
关键函数与参数说明
1. 核心增强函数
wool_specific_augmentation()
返回轻 / 中 / 重三档基础增强管道，适配不同数据需求：
轻度增强（light）：低概率变换（如 30% 水平翻转、20% 亮度调整），适合数据量充足但需少量扩充的场景。
中度增强（moderate）：中等概率变换（如 50% 翻转、30% 颜色调整），平衡多样性与特征保留。
重度增强（heavy）：高概率复杂变换（如 70% 翻转、60% 亮度调整、弹性变换），适合数据量极少的场景。
wool_fiber_aware_augmentation()
专为细羊毛设计，限制旋转角度（≤15°）、缩小仿射变换幅度，避免破坏纤维结构特征。
wool_quality_augmentation()
强化羊毛质量相关特征（如纹理锐化、对比度提升），适合羊毛缺陷检测、细度分级等任务。
wool_environment_augmentation()
模拟真实拍摄环境变化，包含雨天（RandomRain）、雪天（RandomSnow）、光照波动（RandomGamma）等效果。
2. 图像处理函数
process_wool_images(...)
一次增强主函数，参数说明：
参数名	类型	含义与可选值
input_folder	str	原始图像文件夹绝对路径
output_folder	str	增强图像输出文件夹绝对路径（自动创建）
augmentation_strength	str	增强强度：light/moderate/heavy
target_multiplier	int	目标总增强图像数（按原始图像数量均分生成量）
wool_type	str	羊毛类型：fine（细羊毛）/coarse（粗羊毛）/None（通用）
second_environment_process(...)
二次环境增强函数，参数说明：
参数名	类型	含义
input_folder	str	一次增强结果文件夹路径（作为二次输入）
output_folder	str	二次增强输出文件夹路径
target_multiplier	int	二次增强目标总图像数
输出说明
图像命名规则：增强后的图像命名格式为 原始文件名_wool_aug_序号.jpg，例如 wool_sample1_wool_aug_3.jpg（原始文件 wool_sample1.jpg 的第 3 张增强图）。
图像质量：采用 JPEG 格式保存，质量参数为 95（cv2.IMWRITE_JPEG_QUALITY, 95），平衡清晰度与文件大小。
格式转换：内部自动完成 BGR（OpenCV默认）→ RGB（增强处理）→ BGR（保存） 的格式转换，确保增强效果正确。
注意事项
路径问题：必须使用绝对路径（如 D:/Galaxy/wool_data），相对路径可能导致文件找不到；避免中文 / 特殊字符路径。
图像读取失败：若日志提示 “无法读取图像”，检查：
图像文件是否损坏。
图像格式是否在支持列表（jpg/jpeg/png/bmp/tiff）。
增强强度选择：
数据量充足：优先选择 light/moderate，避免过度增强导致特征失真。
数据量极少：选择 heavy + 二次增强，最大化数据多样性。
羊毛类型适配：
细羊毛（纤维细、结构敏感）：设置 wool_type="fine"，自动启用纤维感知增强。
粗羊毛（结构健壮）：设置 wool_type="coarse"，可使用 heavy 强度增强。
常见问题
Q1：报错 ModuleNotFoundError: No module named 'albumentations'？
A1：未安装依赖库，执行以下命令重新安装：
bash
pip install albumentations -i https://pypi.tuna.tsinghua.edu.cn/simple/
Q2：增强后的图像数量与 target_multiplier 不一致？
A2：target_multiplier 是总目标增强数，工具会按原始图像数量均分生成量（如 20 张原始图 + 目标 50 张，每张生成 2~3 张），确保总数量接近目标值。
Q3：环境增强没有雨雪效果？
A3：RandomRain/RandomSnow 的概率为 30%（p=0.3），并非每张图都会添加效果，可修改 wool_environment_augmentation() 中的 p 值提升概率（如 p=0.5）。