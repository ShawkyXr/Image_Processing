import tkinter as tk
import os

def open_image_classification_app():
    os.system('python3 Image_Classification_App/main.py')

root = tk.Tk()
root.geometry("600x400")
root.title('Image Processing App')
root.configure(bg='SaddleBrown')

header = tk.Label(root, text='Image Classification App', font=('JetBrains Mono', 28, 'bold'), bg='SaddleBrown', fg='white')
header.pack(side=tk.TOP, fill=tk.X, pady=80)


classify_btn = tk.Button(root, text='Image Classification',width=30 , height= 2 , font=('JetBrains Mono', 16, 'bold'), fg='black', command = open_image_classification_app)
classify_btn.pack(pady=30)


if __name__ == "__main__":
    root.mainloop()