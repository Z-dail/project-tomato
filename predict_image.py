import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import pandas as pd
import json
import glob
import random


def load_model(model_path):
    """加载训练好的模型"""
    print(f"加载模型: {model_path}")

    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在! {model_path}")
        # 尝试查找其他模型文件
        model_dir = os.path.dirname(model_path)
        if os.path.exists(model_dir):
            models = [f for f in os.listdir(model_dir) if f.endswith('.h5')]
            print(f"可用模型文件: {models}")
            if models:
                model_path = os.path.join(model_dir, models[0])
                print(f"使用模型: {model_path}")
            else:
                return None

    try:
        model = keras.models.load_model(model_path)
        print(f"模型加载成功!")
        print(f"模型输入形状: {model.input_shape}")
        print(f"模型输出形状: {model.output_shape}")
        return model
    except Exception as e:
        print(f"模型加载失败: {e}")
        return None


def load_class_mapping(mapping_path="PlantVillage-Tomato/class_mapping.json"):
    """加载类别映射"""
    if os.path.exists(mapping_path):
        with open(mapping_path, 'r') as f:
            class_mapping = json.load(f)
        # 将字符串键转换为整数键
        class_mapping = {int(k): v for k, v in class_mapping.items()}
        print(f"加载类别映射: {len(class_mapping)} 个类别")
        return class_mapping
    else:
        print(f"警告: 类别映射文件不存在: {mapping_path}")
        return None


def preprocess_image(img_path, target_size=(224, 224)):
    """预处理图像"""
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # 添加批次维度
    img_array = img_array / 255.0  # 归一化
    return img_array


def predict_single_image(model, img_path, class_mapping=None, threshold=0.5):
    """预测单张图像"""
    print(f"\n预测图像: {os.path.basename(img_path)}")

    # 检查文件是否存在
    if not os.path.exists(img_path):
        print(f"错误: 图像文件不存在! {img_path}")
        return None, None

    # 预处理图像
    img_array = preprocess_image(img_path, target_size=model.input_shape[1:3])

    # 进行预测
    predictions = model.predict(img_array, verbose=0)
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0])

    # 获取类别名称
    if class_mapping:
        class_name = class_mapping.get(predicted_class, f"类别{predicted_class}")
    else:
        class_name = f"类别{predicted_class}"

    print(f"预测结果: {class_name} (置信度: {confidence:.2%})")

    # 显示所有类别概率
    print("所有类别概率:")
    for i, prob in enumerate(predictions[0]):
        if class_mapping:
            name = class_mapping.get(i, f"类别{i}")
        else:
            name = f"类别{i}"
        if prob > 0.01:  # 只显示概率大于1%的类别
            print(f"  {name}: {prob:.2%}")

    return predicted_class, confidence


def predict_multiple_images(model, image_dir, class_mapping=None, num_images=9):
    """预测并显示多张图像"""
    # 获取所有图像文件
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_files = []

    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(image_dir, ext)))
        image_files.extend(glob.glob(os.path.join(image_dir, '**', ext), recursive=True))

    if not image_files:
        print(f"在 {image_dir} 中没有找到图像文件")
        return

    print(f"找到 {len(image_files)} 个图像文件")

    # 随机选择指定数量的图像
    if len(image_files) > num_images:
        selected_images = random.sample(image_files, num_images)
    else:
        selected_images = image_files
        num_images = len(image_files)

    # 设置子图
    rows = int(np.ceil(num_images / 3))
    cols = min(num_images, 3)
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))

    if num_images == 1:
        axes = np.array([axes])

    axes = axes.flatten() if num_images > 1 else [axes]

    # 预测并显示每张图像
    for idx, (ax, img_path) in enumerate(zip(axes[:num_images], selected_images)):
        try:
            # 预测
            img_array = preprocess_image(img_path, target_size=model.input_shape[1:3])
            predictions = model.predict(img_array, verbose=0)
            predicted_class = np.argmax(predictions[0])
            confidence = np.max(predictions[0])

            # 获取类别名称
            if class_mapping:
                class_name = class_mapping.get(predicted_class, f"类别{predicted_class}")
            else:
                class_name = f"类别{predicted_class}"

            # 显示图像
            img = mpimg.imread(img_path)
            ax.imshow(img)
            ax.set_title(f"{class_name}\n置信度: {confidence:.2%}", fontsize=12)
            ax.axis('off')

            # 在图像文件名中添加预测结果
            filename = os.path.basename(img_path)
            print(f"{idx + 1}. {filename} -> {class_name} ({confidence:.2%})")

        except Exception as e:
            print(f"处理图像 {img_path} 时出错: {e}")
            ax.axis('off')
            ax.set_title("处理失败")

    # 隐藏多余的子图
    for ax in axes[num_images:]:
        ax.axis('off')

    plt.tight_layout()
    plt.suptitle(f"图像预测结果 (共{num_images}张)", fontsize=16, y=1.02)
    plt.show()


def predict_test_set(model, test_dir, class_mapping=None):
    """预测整个测试集"""
    print(f"\n预测测试集: {test_dir}")

    # 创建数据生成器
    test_datagen = image.ImageDataGenerator(rescale=1. / 255)

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=model.input_shape[1:3],
        batch_size=32,
        shuffle=False,
        class_mode='categorical')

    print(f"测试集样本数: {test_generator.samples}")
    print(f"类别数: {test_generator.num_classes}")

    # 评估模型
    print("\n评估模型在测试集上的性能...")
    evaluation = model.evaluate(test_generator, verbose=1)

    if len(evaluation) >= 2:
        print(f"测试集损失: {evaluation[0]:.4f}")
        print(f"测试集准确率: {evaluation[1]:.2%}")

    # 进行预测
    print("\n进行预测...")
    predictions = model.predict(test_generator, verbose=1)

    # 获取预测类别
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_generator.classes

    # 计算准确率
    accuracy = np.mean(predicted_classes == true_classes)
    print(f"预测准确率: {accuracy:.2%}")

    # 显示混淆矩阵
    from sklearn.metrics import confusion_matrix, classification_report
    import seaborn as sns

    cm = confusion_matrix(true_classes, predicted_classes)

    # 获取类别名称
    if class_mapping:
        class_names = [class_mapping.get(i, f"类别{i}") for i in range(len(class_mapping))]
    else:
        class_names = [f"类别{i}" for i in range(test_generator.num_classes)]

    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('混淆矩阵', fontsize=16)
    plt.xlabel('预测类别', fontsize=14)
    plt.ylabel('真实类别', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    # 显示分类报告
    print("\n分类报告:")
    print(classification_report(true_classes, predicted_classes,
                                target_names=class_names))

    return predictions, predicted_classes, true_classes


def interactive_prediction(model, class_mapping):
    """交互式预测：用户可以输入图像路径"""
    print("\n" + "=" * 50)
    print("交互式图像预测")
    print("=" * 50)
    print("输入图像路径进行预测（输入 'q' 退出）")

    while True:
        img_path = input("\n输入图像路径: ").strip()

        if img_path.lower() == 'q':
            print("退出预测")
            break

        if not os.path.exists(img_path):
            print(f"文件不存在: {img_path}")
            continue

        try:
            # 预测
            predicted_class, confidence = predict_single_image(model, img_path, class_mapping)

            if predicted_class is not None:
                # 显示图像
                plt.figure(figsize=(8, 8))
                img = mpimg.imread(img_path)
                plt.imshow(img)

                if class_mapping:
                    class_name = class_mapping.get(predicted_class, f"类别{predicted_class}")
                else:
                    class_name = f"类别{predicted_class}"

                plt.title(f"预测结果: {class_name}\n置信度: {confidence:.2%}", fontsize=16)
                plt.axis('off')
                plt.tight_layout()
                plt.show()
        except Exception as e:
            print(f"预测失败: {e}")


def main():
    # 配置参数
    MODEL_PATH = "saved_models/MobileNetV1_WithoutCLAHE_NoAug_WithoutDense_ValBest.h5"
    TEST_DIR = "PlantVillage-Tomato/Test"  # 测试集目录
    SAMPLE_IMAGE_DIR = "PlantVillage-Tomato/Test"  # 示例图像目录

    # 1. 加载模型
    model = load_model(MODEL_PATH)
    if model is None:
        print("无法加载模型，程序退出")
        return

    # 2. 加载类别映射
    class_mapping = load_class_mapping()

    # 3. 选择预测模式
    print("\n" + "=" * 50)
    print("选择预测模式:")
    print("1. 预测单张图像")
    print("2. 预测多张图像")
    print("3. 预测整个测试集")
    print("4. 交互式预测")
    print("5. 退出")
    print("=" * 50)

    choice = input("请输入选择 (1-5): ").strip()

    if choice == '1':
        # 预测单张图像
        img_path = input("输入图像路径: ").strip()
        if os.path.exists(img_path):
            predicted_class, confidence = predict_single_image(model, img_path, class_mapping)

            # 显示图像
            if predicted_class is not None:
                plt.figure(figsize=(8, 8))
                img = mpimg.imread(img_path)
                plt.imshow(img)

                if class_mapping:
                    class_name = class_mapping.get(predicted_class, f"类别{predicted_class}")
                else:
                    class_name = f"类别{predicted_class}"

                plt.title(f"预测结果: {class_name}\n置信度: {confidence:.2%}", fontsize=16)
                plt.axis('off')
                plt.tight_layout()
                plt.show()
        else:
            print(f"文件不存在: {img_path}")

    elif choice == '2':
        # 预测多张图像
        img_dir = input(f"输入图像目录 [按Enter使用默认: {SAMPLE_IMAGE_DIR}]: ").strip()
        if not img_dir:
            img_dir = SAMPLE_IMAGE_DIR

        num_images = input("输入要显示的图像数量 [按Enter使用默认: 9]: ").strip()
        if num_images:
            num_images = int(num_images)
        else:
            num_images = 9

        predict_multiple_images(model, img_dir, class_mapping, num_images)

    elif choice == '3':
        # 预测整个测试集
        test_dir = input(f"输入测试集目录 [按Enter使用默认: {TEST_DIR}]: ").strip()
        if not test_dir:
            test_dir = TEST_DIR

        if os.path.exists(test_dir):
            predict_test_set(model, test_dir, class_mapping)
        else:
            print(f"测试集目录不存在: {test_dir}")

    elif choice == '4':
        # 交互式预测
        interactive_prediction(model, class_mapping)

    elif choice == '5':
        print("退出程序")

    else:
        print("无效选择")


if __name__ == "__main__":
    main()