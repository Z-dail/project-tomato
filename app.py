from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import os
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from enhancements import applyCLAHE
import json
import tempfile
import logging
from werkzeug.utils import secure_filename
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # 启用CORS支持

# 远程服务器配置
PORT = int(os.environ.get('PORT', 14120))  # 使用环境变量PORT或默认14120
HOST = '0.0.0.0'  # 绑定到所有网络接口
DEBUG = os.environ.get('DEBUG', 'False').lower() == 'true'

# 配置
MODEL_PATH = "saved_models/MobileNetV2_WithCLAHE_NoAug_WithoutDense_ValBest.h5"
CLASS_MAPPING_PATH = "PlantVillage-Tomato/class_mapping.json"
UPLOAD_FOLDER = tempfile.mkdtemp()
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 限制上传文件大小为16MB

# 全局变量
model = None
class_mapping = None


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_model_and_mapping():
    """加载模型和类别映射"""
    global model, class_mapping

    try:
        logger.info(f"开始加载模型: {MODEL_PATH}")

        # 检查模型文件是否存在
        if not os.path.exists(MODEL_PATH):
            logger.error(f"模型文件不存在: {MODEL_PATH}")
            return False

        # 加载模型
        model = keras.models.load_model(MODEL_PATH)
        logger.info(f"模型加载成功! 输入形状: {model.input_shape}")

        # 加载类别映射
        if os.path.exists(CLASS_MAPPING_PATH):
            with open(CLASS_MAPPING_PATH, 'r', encoding='utf-8') as f:
                class_mapping = json.load(f)
                # 转换键为整数
                class_mapping = {int(k): v for k, v in class_mapping.items()}
                logger.info(f"类别映射加载成功: {len(class_mapping)} 个类别")
                logger.info(f"类别映射: {class_mapping}")
        else:
            # 如果没有映射文件，创建默认映射
            class_mapping = {i: f"病害类别{i}" for i in range(10)}
            logger.warning("类别映射文件不存在，使用默认映射")

        return True

    except Exception as e:
        logger.error(f"加载模型失败: {e}")
        return False


def preprocess_image_with_clahe(img_path, target_size=(224, 224)):
    """使用CLAHE预处理图像"""
    try:
        # 加载图像
        img = image.load_img(img_path, target_size=target_size)
        img_array = image.img_to_array(img)

        # 应用CLAHE增强
        img_array = applyCLAHE(img_array.astype(np.uint8))

        # 添加批次维度并归一化
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        return img_array
    except Exception as e:
        logger.error(f"图像预处理失败: {e}")
        return None


@app.route('/')
def index():
    """主页"""
    try:
        with open('predict.html', 'r', encoding='utf-8') as f:
            return render_template_string(f.read())
    except Exception as e:
        logger.error(f"读取HTML文件失败: {e}")
        return f"""
        <html>
        <head><title>番茄病害预测系统</title></head>
        <body>
        <h1>番茄病害预测系统</h1>
        <p>系统暂时不可用，请稍后重试。</p>
        <p>错误信息: {str(e)}</p>
        </body>
        </html>
        """


@app.route('/predict', methods=['POST'])
def predict():
    """预测接口"""
    start_time = datetime.now()

    try:
        logger.info("收到预测请求")

        if 'file' not in request.files:
            logger.warning("请求中没有文件")
            return jsonify({'success': False, 'error': '没有上传文件'})

        file = request.files['file']
        if file.filename == '':
            logger.warning("文件名为空")
            return jsonify({'success': False, 'error': '文件名为空'})

        if not allowed_file(file.filename):
            logger.warning(f"不支持的文件类型: {file.filename}")
            return jsonify({'success': False, 'error': '不支持的文件类型'})

        # 保存上传的文件
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        logger.info(f"文件保存成功: {filepath}")

        # 预处理图像（使用CLAHE）
        img_array = preprocess_image_with_clahe(filepath)
        if img_array is None:
            logger.error("图像预处理失败")
            return jsonify({'success': False, 'error': '图像预处理失败'})

        # 进行预测
        logger.info("开始模型预测")
        predictions = model.predict(img_array, verbose=0)
        predicted_class = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))

        logger.info(f"预测完成 - 类别: {predicted_class}, 置信度: {confidence:.4f}")

        # 获取类别名称
        class_name = class_mapping.get(predicted_class, f"病害类别{predicted_class}")

        # 获取Top-3预测结果
        top3_indices = np.argsort(predictions[0])[-3:][::-1]
        top3 = []
        for idx in top3_indices:
            top3.append({
                'class': class_mapping.get(idx, f"病害类别{idx}"),
                'confidence': float(predictions[0][idx])
            })

        # 清理临时文件
        try:
            os.remove(filepath)
            logger.info(f"临时文件清理成功: {filepath}")
        except Exception as e:
            logger.warning(f"清理临时文件失败: {e}")

        # 计算处理时间
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"预测请求处理完成，耗时: {processing_time:.2f}秒")

        return jsonify({
            'success': True,
            'prediction': class_name,
            'confidence': confidence,
            'top3': top3,
            'processing_time': processing_time
        })

    except Exception as e:
        logger.error(f"预测过程出错: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)})


@app.route('/health')
def health():
    """健康检查接口"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat(),
        'server_info': {
            'host': HOST,
            'port': PORT,
            'debug': DEBUG
        }
    })


@app.route('/info')
def info():
    """系统信息接口"""
    return jsonify({
        'model_path': MODEL_PATH,
        'class_mapping_path': CLASS_MAPPING_PATH,
        'model_loaded': model is not None,
        'class_mapping': class_mapping,
        'server_time': datetime.now().isoformat()
    })


@app.errorhandler(413)
def too_large(e):
    return jsonify({'success': False, 'error': '文件过大，请上传小于16MB的文件'}), 413


@app.errorhandler(500)
def internal_error(e):
    logger.error(f"服务器内部错误: {e}", exc_info=True)
    return jsonify({'success': False, 'error': '服务器内部错误'}), 500


# 在app.py末尾修改
if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("番茄病害预测系统启动中...")
    logger.info(f"服务器配置: HOST={HOST}, PORT={PORT}, DEBUG={DEBUG}")
    logger.info("警告: 当前使用Flask开发服务器，建议生产环境使用Gunicorn")

    # 启动时加载模型
    if load_model_and_mapping():
        logger.info("模型加载成功，启动Web服务...")
        try:
            # 生产环境优化配置
            app.run(
                host=HOST,
                port=PORT,
                debug=DEBUG,
                threaded=True,  # 启用多线程
                use_reloader=False  # 禁用重载器
            )
        except Exception as e:
            logger.error(f"服务器启动失败: {e}")
    else:
        logger.error("模型加载失败，无法启动服务")