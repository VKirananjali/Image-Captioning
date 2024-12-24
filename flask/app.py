import os
import numpy as np

from pickle import load
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.preprocessing.image import load_img, img_to_array

from flask import Flask , request, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)

model=load_model("resnet.h5")
with open('tokenizer.pkl', 'rb') as file:
    tokenizer = load(file)
caption_model= load_model('img_caption_model_Attention.keras')

@app.route('/')
def home():
    return render_template("index.html")

def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=31, padding='post', truncating='post')
        yhat = model.predict([photo,sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text

@app.route('/predict',methods=["POST"])
def predict():
    file = request.files['image']
    filename = secure_filename(file.filename)
    basepath=os.path.dirname(__file__)
    print('current path : ', basepath)

    filepath= os.path.join(basepath,'static/uploads',filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    print('file path : ', filepath)
    file.save(filepath)

    img = load_img(filepath, target_size=(224,224))
    x=img_to_array(img)
    x=np.expand_dims(x,axis=0)
    x=preprocess_input(x)
    image = model.predict(x, verbose=0)
    print("preprocessed image")
    description = generate_desc(caption_model, tokenizer, image, 30)
    print(description)
    return render_template("index.html",result=description)

if __name__=="__main__":
    app.run(debug=False, threaded=False)