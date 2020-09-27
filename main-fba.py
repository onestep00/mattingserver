import io
import cv2
import os
import sys
import argparse
from flask import Flask, request, send_file, jsonify
from flask_ngrok import run_with_ngrok
from PIL import Image
import numpy as np
import time
import logging
import uuid

from u2net import U2net
# from gca import Gca
from fba import Fba

parser = argparse.ArgumentParser()
parser.add_argument('--u2', type=str, default='./saved_models/u2net/u2net.pth')
parser.add_argument('--u2-size', type=str, default='320')
parser.add_argument('--fba', type=str, default='./saved_models/gca/gca-dist-all-data.pth')

# Parse configuration
args = parser.parse_args()

logging.basicConfig(level=logging.INFO)

original_path = './images/original/'
mask_path = './images/mask/'
cut_path = './images/cut/'

app = Flask(__name__)
run_with_ngrok(app)

u2net = U2net(args.u2, args.u2_size)
gca = Gca(args.gca, args.gca_config)

@app.route('/', methods=['POST'])
def save():
    start = time.time()
    filename = uuid.uuid4().hex + '.png'

    # Convert string of image data to uint8
    if 'data' not in request.files:
        return jsonify({'error': 'missing file param `data`'}), 400
    data = request.files['data'].read()
    if len(data) == 0:
        return jsonify({'error': 'empty image'}), 400

    # Convert string data to PIL Image
    img = Image.open(io.BytesIO(data))
    img.save(original_path + filename, 'PNG')

    # Ensure i,qge size is under 1024
    if img.size[0] > 1024 or img.size[1] > 1024:
        img.thumbnail((1024, 1024))

    np_img = np.array(img)
    res = u2net.run(np_img)
    res[res<125] = 0
    res[res>170] = 255
    # Process Image
    # kernel_er = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # kernel_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # rcnn=rcnn.astype(np.float32)/255; rcnn[rcnn>0.2]=1;
    # K=25

    # zero_id=np.nonzero(np.sum(rcnn,axis=1)==0)
    # del_id=zero_id[0][zero_id[0]>250]
    # if len(del_id)>0:
    #   del_id=[del_id[0]-2,del_id[0]-1,*del_id]
    #   rcnn=np.delete(rcnn,del_id,0)
    # rcnn = cv2.copyMakeBorder( rcnn, 0, K + len(del_id), 0, 0, cv2.BORDER_REPLICATE)


    # rcnn = cv2.erode(rcnn, kernel_er, iterations=10)
    # rcnn = cv2.dilate(rcnn, kernel_dil, iterations=5)
    # rcnn=cv2.GaussianBlur(rcnn.astype(np.float32),(31,31),0)
    # rcnn=(255*rcnn).astype(np.uint8)
    # rcnn=np.delete(rcnn, range(reso[0],reso[0]+K), 0)
    
    # res_img = res.convert('L')
    # res_img = res_img.resize(img.size)
    res_img = res

    kernel_er = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    res_img1 = cv2.erode(res_img, kernel_er, iterations=10)
    res_img2 = cv2.dilate(res_img, kernel_dil, iterations=5)
    # res_img=cv2.GaussianBlur(res_img,(31,31),0)
    # res_img1 = cv2.erode(res_img, kernel, iterations=10)
    # res_img2 = cv2.dilate(res_img, kernel, iterations=15)

    res_img = res_img1/2 + res_img2/2

    # res_img[[res_img1 < 200] and [res_img2 > 125]] = 125

    # res_img = cv2.GaussianBlur(res_img,(71,71),cv2.BORDER_DEFAULT)
    # res_img = cv2.GaussianBlur(res_img,(71,71),cv2.BORDER_DEFAULT)
    # res_img = cv2.GaussianBlur(res_img,(71,71),cv2.BORDER_DEFAULT)
    # res_img = cv2.GaussianBlur(res_img,(71,71),cv2.BORDER_DEFAULT)
    res_img = cv2.resize(res_img, (np_img.shape[1], np_img.shape[0]))
    # res_img = cv2.resize(res, (np_img.shape[1], np_img.shape[0]))
    # logging.info(res.status_code)

    # Save mask locally.
    logging.info(' > saving results...')
    save_res_img = Image.fromarray(res_img).convert('RGB')
    save_res_img.save(mask_path + filename, 'PNG')

    # Matting.
    mat_img = gca.run(np_img, res_img)
    mat_img = Image.fromarray(mat_img)

    # Convert string data to PIL Image.
    logging.info(' > compositing final image...')
    empty = Image.new("RGBA", img.size, 0)
    cut_img = Image.composite(img, empty, mat_img)

    # TODO: currently hack to manually scale up the images. Ideally this would
    # be done respective to the view distance from the screen.
    # img_scaled = img.resize((img.size[0] * 3, img.size[1] * 3))

    # Save locally.
    logging.info(' > saving final image...')
    # img_scaled.save('cut_current.png')
    cut_img.save(cut_path + filename, 'PNG')

    # Save to buffer
    buff = io.BytesIO()
    cut_img.save(buff, 'PNG')
    buff.seek(0)

    # Print stats
    logging.info(f'Completed in {time.time() - start:.2f}s')

    # Return data
    return send_file(buff, mimetype='image/png')

app.run()