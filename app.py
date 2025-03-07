from flask import Flask, request, Response
from gfpgan import GFPGANer
import cv2
import numpy as np
import os
from io import BytesIO

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results/restored_imgs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# 初始化 GFPGAN
restorer = GFPGANer(
    model_path='experiments/pretrained_models/GFPGANv1.4.pth',
    upscale=2,
    arch='clean',
    channel_multiplier=2
)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "请上传图片", 400
    file = request.files['file']
    if file.filename == '':
        return "未选择图片", 400
    
    # 保存上传的图片
    input_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(input_path)

    # 读取并处理图片
    img = cv2.imread(input_path)
    if img is None:
        return "无法读取图片", 500

    _, _, restored_img = restorer.enhance(img)
    output_path = os.path.join(RESULT_FOLDER, f'restored_{file.filename}')
    cv2.imwrite(output_path, restored_img)

    # 返回修复后的图片字节流
    _, buffer = cv2.imencode('.jpg', restored_img)
    img_bytes = BytesIO(buffer.tobytes())
    return Response(img_bytes.getvalue(), mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))