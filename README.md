# EX 10 Implementation of SVM For Spam Mail Detection
## DATE:
## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
 1. Data Preprocessing
 2. Feature Extraction
 3. Model Training
 4. Model Evaluation
 5. Prediction

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection.
Developed by:Raja rithika
RegisterNumber:  2305001029
*/
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.metrics import classification_report,accuracy_score

df=pd.read_csv('/content/spamEX10.csv',encoding='ISO-8859-1')
df.head()

vectorizer=CountVectorizer()
X=vectorizer.fit_transform(df['v2'])
y=df['v1']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

model=svm.SVC(kernel='linear')
model.fit(X_train,y_train)

predictions=model.predict(X_test)
print("Accuracy:",accuracy_score(y_test,predictions))
print("Classification Report:")
print(classification_report(y_test,predictions))

def predict_message(message):
    message_vec = vectorizer.transform([message])
    prediction = model.predict(message_vec)
    return prediction[0]

new_message="Free prixze money winner"
result=predict_message(new_message)
print(f"The message: '{new_message}' is classified as: {result}")
```

## Output:
![image](https://github.com/user-attachments/assets/636e88a6-6b88-4b33-994d-11c6940c730a)
![image](https://github.com/user-attachments/assets/0e7ee740-3882-490e-a77e-23b38b466e63)
![image](https://github.com/user-attachments/assets/43b8988d-5ce0-462e-9575-561d1dd409d8)





## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
