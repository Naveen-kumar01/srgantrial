import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS, cross_origin
import boto3


os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)

from data import DIV2K
from model.srgan import generator, discriminator
from train import SrganTrainer, SrganGeneratorTrainer

# load pretrained model weights
weights_dir = 'weights/srgan'
weights_file = lambda filename: os.path.join(weights_dir, filename)

os.makedirs(weights_dir, exist_ok=True)

pre_generator = generator()
gan_generator = generator()

pre_generator.load_weights(weights_file('pre_generator.h5'))
gan_generator.load_weights(weights_file('gan_generator.h5'))

# demo code
from model import resolve_single
from utils import load_image

def resolve_and_plot(lr_image_path):
    lr = load_image(lr_image_path)

    pre_sr = resolve_single(pre_generator, lr)
    gan_sr = resolve_single(gan_generator, lr)
    
    plt.figure(figsize=(20, 20))
    
    images = [lr, pre_sr, gan_sr]
    titles = ['LR', 'SR (PRE)', 'SR (GAN)']
    positions = [1, 3, 4]
    
    for i, (img, title, pos) in enumerate(zip(images, titles, positions)):
        plt.subplot(2, 2, pos)
        plt.imshow(img)
        plt.title(title, fontsize=40)
        plt.xticks([])
        plt.yticks([])
        plt.savefig('static/title')

@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('head.html')


@app.route("/result", methods=['GET', 'POST'])
@cross_origin()
def predictRoute():
    if request.method == "POST":
        pred_list = request.form.to_dict()
        image = (pred_list['path'])
        image_path = 'data/'+image
        s3 = boto3.resource(service_name="s3",
                            region_name="us-east-2",
                            aws_access_key_id="AKIA4HV2ZHHGMISVHC5W",
                            aws_secret_access_key="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")
        data = s3.Bucket('aqi-data001')
        data.download_file(image_path, 'image1.jpg')
        resolve_and_plot('image1.jpg')
        return render_template("res.html")


# port = int(os.getenv("PORT"))
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True)





