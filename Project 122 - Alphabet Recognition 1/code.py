import cv2 
import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn. linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image 
import PIL.ImageOps
import os, ssl, time

if(not os.environ.get('PYTHONHTTPSVERIFY','') and getattr(ssl,'_create_unverified_context', None)) :
    ssl._create_default_https_context = ssl._create_unverified_context

X, y = fetch_openml("mnist_784", version = 1, return_X_y = True)
print(pd.Series(y).value_counts())

classes = ['A', 'B', 'C', 'D', 'E','F', 'G', 'H', 'I', 'J', "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
print(len(classes))
nclasses = len(classes)

samples_per_class = 5
figure = plt.figure(figsize=(nclasses*2,(1+samples_per_class*2)))

idx_cls = 0
for cls in classes:
  idxs = np.flatnonzero(y == cls)
  idxs = np.random.choice(idxs, samples_per_class, replace=False)
  i = 0
  for idx in idxs:
    plt_idx = i * nclasses + idx_cls + 1
    p = plt.subplot(samples_per_class, nclasses, plt_idx);
    p = sns.heatmap(np.reshape(X[idx], (22,30)), cmap=plt.cm.gray, 
             xticklabels=False, yticklabels=False, cbar=False);
    p = plt.axis('off');
    i += 1
  idx_cls += 1

x_train, x_test, y_train, y_test = train_test_split(X, y, random_state = 9, train_size = 7500, test_size = 2500)

x_train_scaled = x_train/255.0
x_test_scaled = x_test/255.0

clf = LogisticRegression(solver = "saga", multi_class = "multinomial").fit(x_train_scaled, y_train)
y_prediction = clf.predict(x_test_scaled)
accuracy = accuracy_score(y_test, y_prediction)
print(accuracy)
cm = pd.crosstab(y_test, y_prediction, rownames=['Actual'], colnames=['Predicted'])

p = plt.figure(figsize=(10,10));
p = sns.heatmap(cm, annot=True, fmt="d", cbar=False)


cap =cv2.VideoCapture(0)
while(True) :
    try :
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        upperleft = (int(width/2 - 56), int(height/2 - 56 ))
        bottomright = (int(width/2 +56), int(height/2 + 56 ))
        cv2.rectangle(gray, upperleft, bottomright, (0, 255, 0), 2)
        roi = gray[upperleft[1] : bottomright[1], upperleft[0] : bottomright[0]]
        im_PIL = Image.fromarray(roi)
        image_bw = im_PIL.convert("L")
        image_bw_resized = image_bw.resize((28, 28), Image.ANTIALIAS)
        image_bw_resized_inverted = PIL.ImageOps.invert(image_bw_resized)
        pixelFilter = 20
        minPixel = np.percentile(image_bw_resized_inverted, pixelFilter)
        image_bw_resized_inverted_scaled = np.clip(image_bw_resized_inverted - minPixel, 0, 255)
        maxPixel = np.max(image_bw_resized_inverted)
        image_bw_resized_inverted_scaled = np.asarray(image_bw_resized_inverted_scaled)/maxPixel
        testSample = np.array(image_bw_resized_inverted_scaled).reshape(1, 784)
        testPrediction = clf.predict(testSample)
        print("Predicted class is", testPrediction)
        cv2.im_show("frame", gray)
        if cv2.waitKey(1) & 0xFF == ord("q") :
            break
    except Exception as e :
        pass
cap.release()
cv2.destroyAllWindows 