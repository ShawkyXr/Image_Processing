import cv2
import numpy as np
import matplotlib.pyplot as plt
from Gaussian_Filter import denoise_image as gaussian_filter
from Median_Filter import denoise_image as median_filter
from Average_Filter import denoise_image as average_filter
import tkinter as tk
from tkinter import messagebox as mb

root = tk.Tk()

def get_image(path):
    if not path:
        raise ValueError("The path cannot be empty.")
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"No image found at the path: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def load_filter(img, filter):
    if filter == 'Gaussian':
        denoised_img = gaussian_filter(img)
    elif filter == 'Median':
        denoised_img = median_filter(img)
    elif filter == 'Average':
        denoised_img = average_filter(img)
    return denoised_img


def show_image(img, original_img):
    # Display the original image
    plt.figure(figsize=(15,10))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(original_img)
    plt.axis("off")

    # Display the denoised image
    plt.subplot(1, 2, 2)
    plt.title("Denoised Image")
    plt.imshow(img)
    plt.axis("off")

    plt.show()


def process_image(path, filter):
    img = get_image(path.get())
    denoised_img = load_filter(img, filter.get())
    show_image(denoised_img,img)

    
def interface():

    #add a heading

    label = tk.Label(root, text='Image Denoising App', font=('JetBrains Mono', 26, 'bold'), bg='SaddleBrown', fg='white')
    label.pack(side=tk.TOP, fill=tk.X, pady=50)

    label = tk.Label(root, text='Enter the path of the image:', font=('JetBrains Mono',16, 'bold'), bg='SaddleBrown', fg='white')
    label.pack()

    path = tk.Entry(root)
    path.pack(pady=10)

    space = tk.Label(root, text='', bg='SaddleBrown')
    space.pack()

    label = tk.Label(root, text='Select the filter:', font=('JetBrains Mono',16, 'bold'), bg='SaddleBrown', fg='white')
    label.pack()


    filter = tk.StringVar()
    filter.set('Gaussian')
    filter_options = ['Gaussian', 'Median', 'Average']
    filter_menu = tk.OptionMenu(root, filter, *filter_options)
    filter_menu.configure(background="white", activebackground="white")

    filter_menu.pack(pady=10)

    def on_process_image():
        try:
            process_image(path, filter)
        except ValueError as ve:
            mb.showerror("Error", str(ve))
        except FileNotFoundError as fnfe:
            mb.showerror("Error", str(fnfe))

    button = tk.Button(root, text='Process Image', command=on_process_image,height=1, width=20, fg='black', font=('JetBrains', 12, 'bold'))
    button.pack(pady=30)



def main():
    root.geometry('500x500')
    root.title('Image Denoising')
    root.configure(bg= 'SaddleBrown')

    interface()

    root.mainloop()

    

if __name__ == '__main__':
    main()