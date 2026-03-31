# import os
# import time
#
# from dataset import load_dataset
# from utils import print_config
# from model import build_model
# import tensorflow as tf
# from tensorflow import keras
# from enhancements import applyCLAHE
#
#
# print("1. 开始加载数据集...")
# train_generator, valid_generator, test_generator = load_dataset("config.json")
# print("\n\n______________CLASS INDICES TO NAME MAPPING_______________")
# print_config(train_generator.class_indices)
# print("___________________________________________________________\n\n")
#
#
# # model = build_model(config_file="inference-config.json")
# # print(model.summary())
#
# print("2. 开始加载模型...")
# # 改为加载V2模型（使用CLAHE增强的模型）
# pretrained_model = keras.models.load_model(os.path.join("saved_models",
#                                                         "MobileNetV1_WithoutCLAHE_NoAug_WithoutDense_ValBest.h5"))
# print(pretrained_model.summary())
#
# # predictions = tf.argmax(pretrained_model.predict(test_generator), axis=1)
#
# print("3. 开始评估模型...")
# result = pretrained_model.evaluate(test_generator)
# print("模型指标名称:", pretrained_model.metrics_names)
# print("评估结果:", result)
#
# print("4. 开始预测...")
# start_time = time.time()
# predictions = pretrained_model.predict(test_generator)
# print(f"预测完成，耗时: {time.time() - start_time:.2f}秒")
# print(f"预测了 {len(predictions)} 个样本")
import os

from dataset import load_dataset
from utils import print_config
from model import build_model
import tensorflow as tf
from tensorflow import keras



train_generator, valid_generator, test_generator = load_dataset()
print("\n\n______________CLASS INDICES TO NAME MAPPING_______________")
print_config(train_generator.class_indices)
print("___________________________________________________________\n\n")


# model = build_model(config_file="inference-config.json")
# print(model.summary())


pretrained_model = keras.models.load_model(os.path.join("saved_models",
                                                                   "MobileNetV1_WithoutCLAHE_NoAug_WithoutDense_ValBest.h5" ))
print(pretrained_model.summary())

# predictions = tf.argmax(pretrained_model.predict(test_generator), axis=1)

result = pretrained_model.evaluate(test_generator)
print(result)
