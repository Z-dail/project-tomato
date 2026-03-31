# -*- coding: utf-8 -*-
import os
import sys
import requests  # 用于下载字体
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# ================= 配置区域 =================
# 1. 强制使用 CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# 2. 模型路径
MODEL_PATH = 'saved_models/MobileNetV1_WithoutCLAHE_NoAug_WithoutDense_ValBest.h5'

# 3. 默认测试图片
DEFAULT_IMG_PATH = 'Septoria_leaf.JPG'

# 4. 类别映射
CLASS_NAMES = {
    0: "健康叶片 (Healthy)",
    1: "早疫病 (Early_blight)",
    2: "晚疫病 (Late_blight)",
    3: "叶霉病 (Leaf_Mold)",
    4: "斑点病 (Septoria_leaf_spot)",
    5: "蜘蛛螨 (Spider_mites)",
    6: "靶斑病 (Target_Spot)",
    7: "花叶病毒 (Mosaic_virus)",
    8: "黄化曲叶病毒 (Yellow_Leaf_Curl_Virus)",
    9: "细菌性斑点病 (Bacterial_spot)"
}


def check_and_download_font():
    """
    检查当前目录下有没有 SimHei.ttf，没有这就自动下载。
    解决所有平台的中文乱码问题。
    """
    font_filename = 'SimHei.ttf'

    if not os.path.exists(font_filename):
        print(f"⚠️ 未检测到中文字体，正在自动下载 {font_filename} ...")
        print("请稍候（约 10MB）...")
        try:
            url = "https://github.com/StellarCN/scp_zh/raw/master/fonts/SimHei.ttf"
            r = requests.get(url)
            with open(font_filename, 'wb') as f:
                f.write(r.content)
            print("✅ 字体下载成功！")
        except Exception as e:
            print(f"❌ 字体下载失败: {e}")
            print("如果下载失败，请手动下载 SimHei.ttf 放入项目文件夹。")
            return None

    return font_filename


def run_prediction(image_path):
    # 1. 准备字体
    font_path = check_and_download_font()
    custom_font = FontProperties(fname=font_path, size=14) if font_path else None

    # 2. 检查文件
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 错误：找不到模型文件 -> {MODEL_PATH}")
        return
    if not os.path.exists(image_path):
        print(f"❌ 错误：找不到图片文件 -> {image_path}")
        return

    # 3. 加载模型
    print("⏳ 正在加载模型...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    # 4. 预处理 (256x256)
    try:
        img = image.load_img(image_path, target_size=(256, 256))
        img_array = image.img_to_array(img) / 255.0
        img_batch = np.expand_dims(img_array, axis=0)
    except Exception as e:
        print(f"❌ 图片处理失败: {e}")
        return

    # 5. 预测
    print(f"🔍 正在分析图片...")
    predictions = model.predict(img_batch)

    # 6. 解析结果
    predicted_id = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    result_text = CLASS_NAMES.get(predicted_id, "未知类别")

    print("=" * 40)
    print(f"✅ 预测结果: {result_text}")
    print(f"📊 置信度:   {confidence:.2%}")
    print("=" * 40)

    # 7. 显示结果
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.axis('off')

    title_str = f"预测结果: {result_text}\n置信度: {confidence:.2%}"

    # 强制使用下载的字体
    if custom_font:
        plt.title(title_str, fontproperties=custom_font, color='blue', fontsize=16)
    else:
        plt.title(title_str, color='blue', fontsize=16)

    print("图片窗口已弹出...")
    plt.show()


if __name__ == "__main__":
    target_img = DEFAULT_IMG_PATH
    if len(sys.argv) > 1:
        target_img = sys.argv[1]
    run_prediction(target_img)