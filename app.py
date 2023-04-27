
from flask import Flask, render_template, request,url_for
import numpy as np
# from tensorflow.keras import load_model 
from PIL import Image
import pickle 

app = Flask(__name__, template_folder='templates')

model = pickle.load(open('modelf.pkl', 'rb'))

def predict(image):
    img = Image.open(image)
    img = img.resize((256,256))
    img = np.array(img)
    img = img.reshape((1, 256, 256, 3))
    img = img/255.0
    pred = model.predict(img)[0]   


    return pred.argmax()

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        image = request.files['image']
        prediction = predict(image)
        return render_template('index.html', prediction=prediction)
    return render_template('index.html')
    

if __name__ == '__main__':
    app.run(debug=True)