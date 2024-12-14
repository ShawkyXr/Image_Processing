from tkinter import *
import tkinter
from PIL import Image, ImageTk
# import all files from this folder
from os import listdir
from os.path import isfile, join
import os
import numpy as np
import matplotlib.pyplot as plt
# import Prediction
# import Model_train

pro = Tk()
pro.geometry("600x600")
pro.title('Image Classification')
pro.configure(bg= 'SaddleBrown')
fr1 = Frame(width = '250'  , height = '250' , bg = 'white')
fr1.place(x = 100 , y = 55)

fr2 = Frame(width = '100'  , height = '30' , bg = 'white')
fr2.place(x = 450 , y = 250)


fr3 = Frame(width = '110'  , height = '30' , bg = 'white')
fr3.place(x = 230, y = 450)

fr4 = Frame(width = '75'  , height = '30' , bg = 'white')
fr4.place(x = 247, y = 400)

bt1 = Button(fr3, text = 'Select Image', bg = 'white', fg = 'black', font = ('JetBrains', 10, 'bold'))
bt1.place(x = 0, y = 0)

bt2 = Button(fr4, text = 'Predict', bg = 'white', fg = 'black', font = ('JetBrains', 10, 'bold'))
bt2.place(x = 0, y = 0)

lbl1 = Label(text = 'Presdiction'  , bg = 'SaddleBrown' , fg = 'white',font = ('JetBrains', 10, 'bold'))
lbl1.place(x = 460 , y = 225)

image_path = "/home/hossam/work/Image_Processing/Image_Classification_App/Dataset/train/cats/cat.4.jpg"
image = Image.open(image_path)
photo = ImageTk.PhotoImage(image)

image_label = Label(fr1, image=photo)
image_label.place(x=0, y=0)
pro.mainloop()
