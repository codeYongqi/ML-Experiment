import tkinter as tk
import tensorflow as tf
from predction_image import predction
from tkinter.filedialog import askopenfilename  
from PIL import Image,ImageTk  

#从文件加载模型
model = tf.keras.models.load_model('my_model.h5')
model.summary()
class_names = ['obj1', 'obj10', 'obj11', 'obj12', 'obj13', 'obj14', 'obj15', 'obj16', 'obj17', 'obj18', 'obj19', 'obj2', 'obj20', 'obj3', 'obj4', 'obj5', 'obj6', 'obj7', 'obj8', 'obj9']
window = tk.Tk()

window.title('My Window')
window.geometry('600x400')

var2= tk.StringVar()

def choosepic():  
    path_=askopenfilename()  
    path.set(path_)  
    res = predction(model,class_names,path_)
    var2.set(res)
    img_open = Image.open(e1.get())  
    img=ImageTk.PhotoImage(img_open) 
    l1.config(image=img)  
    l1.image=img #keep a reference 

path=tk.StringVar()  

button = tk.Button(window,text='选择图片',command=choosepic)
button.pack()

e1=tk.Entry(window,state='readonly',text=path)  
e1.pack()  

l1=tk.Label(window)  
l1.pack()

l = tk.Label(window, textvariable=var2, fg='black', font=('Arial', 12), width=80, height=2)
l.pack()

window.mainloop()


