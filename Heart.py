from subprocess import call
import tkinter as tk
import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageTk
from tkinter import ttk
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

root = tk.Tk()
root.title("Heart Disease")

w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (w, h))
# ++++++++++++++++++++++++++++++++++++++++++++

image2 = Image.open('heart.jpg')

image2 = image2.resize((w, h), Image.ANTIALIAS)

background_image = ImageTk.PhotoImage(image2)


background_label = tk.Label(root, image=background_image)
background_label.image = background_image



background_label.place(x=0, y=0)  # , relwidth=1, relheight=1)
lbl = tk.Label(root, text="Heart Disease Detection System", font=('times', 35,' bold '), height=1, width=32,bg="violet Red",fg="Black")
lbl.place(x=300, y=10)
# _+++++++++++++++++++++++++++++++++++++++++++++++++++++++
data = pd.read_csv("E:/heart_disease_detection/heart.csv")



data = data.dropna()

le = LabelEncoder()
data['AHD'] = le.fit_transform(data['AHD'])

data['Thal'] = le.fit_transform(data['Thal'])
data['ChestPain'] = le.fit_transform(data['ChestPain'])

data.head()

"""Feature Selection => Manual"""
x = data.drop(['AHD', 'Series'], axis=1)


def Data_Preprocessing():
    data = pd.read_csv("E:/heart_disease_detection/heart.csv")
    data.head()

    data = data.dropna()

    """One Hot Encoding"""

    le = LabelEncoder()
    data['AHD'] = le.fit_transform(data['AHD'])
    print(data['Ca'])
    data['Thal'] = le.fit_transform(data['Thal'])
    print("thal Encoding")
    data['ChestPain'] = le.fit_transform(data['ChestPain'])

    data['Thal'] = le.fit_transform(data['Thal'])
    data['ChestPain'] = le.fit_transform(data['ChestPain'])

    """Feature Selection => Manual"""
    x = data.drop(['AHD', 'Series'], axis=1)
    data = data.dropna()

    print(type(x))
    y = data['AHD']
    print(type(y))
    x.shape

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

    

    load = tk.Label(root, font=("Tempus Sans ITC", 15, "bold"), width=50, height=2, background="green",
                    foreground="white", text="Data Loaded=>Splitted into 80% for Training & 20% for Testing")
    load.place(x=200, y=80)


def Model_Training():
    data = pd.read_csv("E:/heart_disease_detection/heart.csv")
    data.head()

    data = data.dropna()

    """One Hot Encoding"""

    le = LabelEncoder()
    data['AHD'] = le.fit_transform(data['AHD'])
    print(data['Ca'])
    data['Thal'] = le.fit_transform(data['Thal'])
    print("thal Encoding")
    data['ChestPain'] = le.fit_transform(data['ChestPain'])

    data['Thal'] = le.fit_transform(data['Thal'])
    data['ChestPain'] = le.fit_transform(data['ChestPain'])

    """Feature Selection => Manual"""
    x = data.drop(['AHD', 'Series'], axis=1)
    data = data.dropna()

    print(type(x))
    y = data['AHD']
    print(type(y))
    x.shape

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

    from sklearn.svm import SVC
    svcclassifier = SVC(kernel='linear')
    svcclassifier.fit(x_train, y_train)

    y_pred = svcclassifier.predict(x_test)
    print(y_pred)

    
    print("=" * 40)
    print("==========")
    print("Classification Report : ",(classification_report(y_test, y_pred)))
    print("Accuracy : ",accuracy_score(y_test,y_pred)*100)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    ACC = (accuracy_score(y_test, y_pred) * 100)
    repo = (classification_report(y_test, y_pred))
    
    label4 = tk.Label(root,text =str(repo),width=45,height=10,bg='khaki',fg='black',font=("Tempus Sanc ITC",14))
    label4.place(x=205,y=200)
    
    label5 = tk.Label(root,text ="Accracy : "+str(ACC)+"%\nModel saved as HEART_DISEASE_MODEL.joblib",width=45,height=3,bg='khaki',fg='black',font=("Tempus Sanc ITC",14))
    label5.place(x=205,y=420)
    from joblib import dump
    dump (svcclassifier,"HEART_DISEASE_MODEL.joblib")
    print("Model saved as HEART_DISEASE_MODEL.joblib")



def call_file():
    import Check_Heart
    Check_Heart.Train()


def Data_Display():
    columns = ['Age', 'Sex', 'Chest Pain', 'Rest BP', 'Chol', 'Fbs', 'Rest ECG', 'Max HR', 'ExAngn', 'OldPeak', 'Slope',
               'Ca']

    data1 = pd.read_csv('heart.csv')

    data1.shape

    data1.shape

    data1.head()

    data1

    data1

    age = data1.ix[:, 1]
    sex = data1.ix[:, 2]
    chest_pain = data1.ix[:, 3]
    rest_bp = data1.ix[:, 4]
    chol = data1.ix[:, 5]
    fbs = data1.ix[:, 6]
    rest_ecg = data1.ix[:, 7]
    max_hr = data1.ix[:, 8]
    exgen = data1.ix[9]
    oldpeak = data1.ix[10]
    slope = data1.ix[11]
    ca = data1.ix[12]

    display = tk.LabelFrame(root, width=100, height=400, )
    display.place(x=200, y=100)

    tree = ttk.Treeview(display, columns=(
    'Age', 'Sex', 'Chest Pain', 'Rest BP', 'Chol', 'Fbs', 'Rest ECG', 'Max HR', 'ExAng', 'OldPeak', 'Slope', 'Ca'))

    style = ttk.Style()
    style.configure('Treeview', rowheight=50)
    style.configure("Treeview.Heading", font=("Tempus Sans ITC", 15, "bold italic"))
    style.configure(".", font=('Helvetica', 15), background="blue")
    style.configure("Treeview", foreground='white', background="black")

    tree["columns"] = ("1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13")
    tree.column("1", width=50)
    tree.column("2", width=50)
    tree.column("3", width=50)
    tree.column("4", width=50)
    tree.column("5", width=50)
    tree.column("6", width=50)
    tree.column("7", width=50)
    tree.column("8", width=50)
    tree.column("9", width=50)
    tree.column("10", width=50)
    tree.column("11", width=50)
    tree.column("12", width=50)
    tree.column("13", width=50)

    tree.heading("1", text="Age")
    tree.heading("2", text="Sex")
    tree.heading("3", text="Chect Pain")
    tree.heading("4", text="Rest BP")
    tree.heading("5", text="Chol")
    tree.heading("6", text="Fbs")
    tree.heading("7", text="Rest ECG")
    tree.heading("8", text="Max Hr")
    tree.heading("9", text="ExAng")
    tree.heading("10", text="old peak")
    tree.heading("11", text="Slope")
    tree.heading("12", text="Ca")

    treeview = tree

    tree.grid(row=0, column=0, sticky=tk.NSEW)

    print("Data Displayed")
    for i in range(0, 304):
        tree.insert("", 'end', values=(
        age[i], sex[i], chest_pain[i], rest_bp[i], chol[i], fbs[i], rest_ecg[i], max_hr[i], exgen[i], oldpeak[i],
        slope[i], ca[i]))
        i = i + 1


check = tk.Frame(root, w=100)
check.place(x=700, y=100)


def window():
    root.destroy()

button2 = tk.Button(root, foreground="white", background="black", font=("Tempus Sans ITC", 14, "bold"),
                    text="Data_Preprocessing", command=Data_Preprocessing, width=15, height=2)
button2.place(x=5, y=90)

button3 = tk.Button(root, foreground="white", background="black", font=("Tempus Sans ITC", 14, "bold"),
                    text="Model Training", command=Model_Training, width=15, height=2)
button3.place(x=5, y=170)

button4 = tk.Button(root, foreground="white", background="black", font=("Tempus Sans ITC", 14, "bold"),
                    text="Disease Detection", command=call_file, width=15, height=2)
button4.place(x=5, y=250)
exit = tk.Button(root, text="Exit", command=window, width=15, height=2, font=('times', 15, ' bold '),bg="red",fg="white")
exit.place(x=5, y=330)

root.mainloop()

'''+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'''