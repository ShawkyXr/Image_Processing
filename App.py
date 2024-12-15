import tkinter as tk
import os

def open_denoising_app():
    os.system('python3 Image_Denoising_App/main.py')

def open_image_classification_app():
    os.system('python3 Image_Classification_App/main.py')

root = tk.Tk()
root.geometry("600x600")
root.title('Image Processing App')
root.configure(bg='SaddleBrown')

header = tk.Label(root, text='Image Processing App', font=('JetBrains Mono', 28, 'bold'), bg='SaddleBrown', fg='white')
header.pack(side=tk.TOP, fill=tk.X, pady=80)

denoise_btn = tk.Button(root, text='Image Denoising',width= 30 ,height= 2 , font=('JetBrains Mono', 16, 'bold'), fg='black', command = open_denoising_app)
denoise_btn.pack(pady=30)


classify_btn = tk.Button(root, text='Image Classification',width=30 , height= 2 , font=('JetBrains Mono', 16, 'bold'), fg='black', command = open_image_classification_app)
classify_btn.pack(pady=30)


if __name__ == "__main__":
    root.mainloop()