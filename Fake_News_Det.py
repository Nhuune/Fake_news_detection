from platform import node
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
#tip can dua tren noi dung
#gom ma 'Tieu De' 'Noi dung' 'nhan'
#Dau va la cac bai bao hoac noi dung dau ra cua du lieu la cac nhan co gia tri 01
#Moi truong la tin tuc chinh tri

#Loading Flask and assigning the model variable
app = Flask(__name__)

tfvect = TfidfVectorizer(stop_words='english', max_df=0.7)
loaded_model = pickle.load(open('model.pkl', 'rb'))
#Importing the cleaned file containing the text and label
dataframe = pd.read_csv('fake_or_real_news.csv')
x = dataframe['text']
y = dataframe['label']
#Splitting the data into train
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

def fake_news_det(news):
    '''
    Học từ vựng và idf, trả về ma trận thuật ngữ tài liệu. 
    idf là truy vấn tìm các x nhập vào có trong dữ liệu hay không
    '''
    tfid_x_train = tfvect.fit_transform(x_train)
    
    	
    #Chuyển đổi tài liệu sang ma trận kỳ hạn tài liệu.
    tfid_x_test = tfvect.transform(x_test)
    
    input_data = [news]
    #Chuyển đổi tài liệu sang ma trận kỳ hạn tài liệu.
    vectorized_input_data = tfvect.transform(input_data)
    prediction = loaded_model.predict(vectorized_input_data)
    return prediction

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        pred = fake_news_det(message)
        print(pred)
        return render_template('index.html', prediction=pred)
    else:
        return render_template('index.html', prediction="Something went wrong")

if __name__ == '__main__':
    app.run(debug=True)