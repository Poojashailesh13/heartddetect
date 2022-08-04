from tkinter import *
def Train():
    """GUI"""
    import tkinter as tk
    import numpy as np
    import pandas as pd

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import LabelEncoder

    root = tk.Tk()

    root.geometry("800x850+250+5")
    root.title("Check Heart Disease")
    root.configure(background="purple")
    
    age = tk.IntVar()
    sex = tk.IntVar()
    chest_pain = tk.IntVar()
    restbp = tk.IntVar()
    chol = tk.IntVar()
    fbs = tk.IntVar()
    restecg = tk.IntVar()
    maxhr = tk.IntVar()
    exang = tk.IntVar()
    oldpeak = tk.DoubleVar()
    slope = tk.IntVar()
    ca = tk.IntVar()
    thal = tk.IntVar()
    
    #===================================================================================================================



    def Detect():
        e1=age.get()
        print(e1)
        e2=sex.get()
        print(e2)
        #b1=Lb1.get(Lb1.curselection())
        #e3.set(b1) 
        #value = Lb1.get(Lb1.curselection())
        #e3.set(value)  
        e3=chest_pain.get()
        print(e3)
        #print(type(e3))
        e4=restbp.get()
        print(e4)
        e5=chol.get()
        print(e5)
        e6=fbs.get()
        print(e6)
        e7=restecg.get()
        print(e7)
        e8=maxhr.get()
        print(e8)
        e9=exang.get()
        print(e9)
        e10=oldpeak.get()
        print(e10)
        e11=slope.get()
        print(e11)
        e12=ca.get()
        print(e12)
        e13=thal.get()
        print(e13)
        #########################################################################################
        
        from joblib import dump , load
        a1=load('E:/heart_disease_detection/HEART_DISEASE_MODEL.joblib')
        v= a1.predict([[e1, e2, e3, e4, e5, e6, e7, e8, e9,e10, e11, e12, e13]])
        print(v)
        if v[0]==1:
            print("Yes")
            yes = tk.Label(root,text="Disease \nDetected!\nReport is Generated",background="red",foreground="white",font=('times', 20, ' bold '),width=15)
            yes.place(x=300,y=100)
            file = open(r"E:\heart_disease_detection\Report.txt", 'w')
            file.write("-----Patient Report-----\n As per input data and system model Heart Disease Detected for Respective Paptient."
                       "\n***Kindly Follow Medicatins***"
                    
                    )
            file.close()
            
        else:
            print("No")
            no = tk.Label(root, text="No Disease \nDetected", background="green", foreground="white",font=('times', 20, ' bold '),width=15)
            no.place(x=300, y=100)
            file = open(r"E:\heart_disease_detection\Report.txt", 'w')
            file.write("-----Patient Report-----\n As per input data and system model No Heart Disease Detected for Respective Paptient."
                       "\n\n***Relax and Follow below mentioned Life Style to be Healthy as You Are!!!***"
                    
                    )
            file.close()



    l1=tk.Label(root,text="Age",background="purple",font=('times', 20, ' bold '),width=10)
    l1.place(x=5,y=1)
    age=tk.Entry(root,bd=2,width=5,font=("TkDefaultFont", 20),textvar=age)
    age.place(x=200,y=1)

    l2=tk.Label(root,text="Sex",background="purple",font=('times', 20, ' bold '),width=10)
    l2.place(x=5,y=50)
    sex=tk.Entry(root,bd=2,width=5,font=("TkDefaultFont", 20),textvar=sex)
    sex.place(x=200,y=50)

    l3=tk.Label(root,text="Chest Pain",background="purple",font=('times', 20, ' bold '),width=10)
    l3.place(x=5,y=100)
    #chest_pain=tk.Entry(root,bd=2,width=5,font=("TkDefaultFont", 20),textvar=chest_pain)
    #chest_pain.place(x=200,y=100)
    
    
    #Lb1 = Listbox(root,width=20,height=3)
    #Lb1.place(x=200,y=100)
    #Lb1.insert(1, "1")
    #Lb1.insert(2, "2")
    #Lb1.insert(3, "3")
    #chest_pain=Lb1.curselection()
    #Lb1.pack()
    R1 = Radiobutton(root, text="Typical", variable=chest_pain, value=1).place(x=200,y=100)
    R2 = Radiobutton(root, text="asymptomatic", variable=chest_pain, value=2).place(x=200,y=120)
    R3 = Radiobutton(root, text="nontypical", variable=chest_pain, value=3).place(x=200,y=140)

    l4=tk.Label(root,text="RestBP",background="purple",font=('times', 20, ' bold '),width=10)
    l4.place(x=5,y=150)
    restbp=tk.Entry(root,bd=2,width=5,font=("TkDefaultFont", 20),textvar=restbp)
    restbp.place(x=200,y=160)

    l5=tk.Label(root,text="Chol",background="purple",font=('times', 20, ' bold '),width=10)
    l5.place(x=5,y=200)
    chol=tk.Entry(root,bd=2,width=5,font=("TkDefaultFont", 20),textvar=chol)
    chol.place(x=200,y=200)

    l6=tk.Label(root,text="FBS",background="purple",font=('times', 20, ' bold '),width=10)
    l6.place(x=5,y=250)
    fbs=tk.Entry(root,bd=2,width=5,font=("TkDefaultFont", 20))
    fbs.place(x=200,y=250)

    l7=tk.Label(root,text="RestECG",background="purple",font=('times', 20, ' bold '),width=10)
    l7.place(x=5,y=300)
    restecg=tk.Entry(root,bd=2,width=5,font=("TkDefaultFont", 20),textvar=restecg)
    restecg.place(x=200,y=300)

    l8=tk.Label(root,text="MaxHR",background="purple",font=('times', 20, ' bold '),width=10)
    l8.place(x=5,y=350)
    maxhr=tk.Entry(root,bd=2,width=5,font=("TkDefaultFont", 20),textvar=maxhr)
    maxhr.place(x=200,y=350)

    l9=tk.Label(root,text="ExANG",background="purple",font=('times', 20, ' bold '),width=10)
    l9.place(x=5,y=400)
    exang=tk.Entry(root,bd=2,width=5,font=("TkDefaultFont", 20),textvar=exang)
    exang.place(x=200,y=400)

    l10=tk.Label(root,text="Old Peak",background="purple",font=('times', 20, ' bold '),width=10)
    l10.place(x=5,y=450)
    oldpeak=tk.Entry(root,bd=2,width=5,font=("TkDefaultFont", 20),textvar=oldpeak)
    oldpeak.place(x=200,y=450)

    l11=tk.Label(root,text="Slope",background="purple",font=('times', 20, ' bold '),width=10)
    l11.place(x=5,y=500)
    slope=tk.Entry(root,bd=2,width=5,font=("TkDefaultFont", 20),textvar=slope)
    slope.place(x=200,y=500)

    l12=tk.Label(root,text="Ca",background="purple",font=('times', 20, ' bold '),width=10)
    l12.place(x=5,y=550)
    ca=tk.Entry(root,bd=2,width=5,font=("TkDefaultFont", 20),textvar=ca)
    ca.place(x=200,y=550)

    l13=tk.Label(root,text="Thal",background="purple",font=('times', 20, ' bold '),width=10)
    l13.place(x=5,y=600)
    #thal=tk.Entry(root,bd=2,width=5,font=("TkDefaultFont", 20),textvar=thal)
    #thal.place(x=200,y=600)
    R4 = Radiobutton(root, text="Fixed", variable=thal, value=1).place(x=200,y=600)
    R5 = Radiobutton(root, text="normal", variable=thal, value=2).place(x=200,y=620)
    R6 = Radiobutton(root, text="reversable", variable=thal, value=3).place(x=200,y=640)

    button1 = tk.Button(root,text="Submit",command=Detect,font=('times', 20, ' bold '),width=10)
    button1.place(x=300,y=10)


    root.mainloop()

Train()