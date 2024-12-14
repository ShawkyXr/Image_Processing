import cv2
import numpy as np
import matplotlib.pyplot as plt
from Gaussian_Filter import denoise_image as gaussian_filter
from Median_Filter import denoise_image as median_filter
from Average_Filter import denoise_image as average_filter
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox as mb

root = tk.Tk()

def get_image(path):
    if not path:
        raise ValueError("Select an image first")
    img = cv2.imread(path)
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

    label = tk.Label(root, text='Image Denoising App', font=('JetBrains Mono', 28, 'bold'), bg='SaddleBrown', fg='white')
    label.pack(side=tk.TOP, fill=tk.X, pady=50)

    label = tk.Label(root, text='select image from your files', font=('JetBrains Mono',16, ), bg='SaddleBrown', fg='white')
    label.pack()

    # Add a button to select the image
    path = tk.StringVar()
    path.set('')

    def on_select_image():
        file_path = tk.filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")])
        path.set(file_path)
        if path.get():
            mb.showinfo("Success", "Image selected successfully")
        else:
            mb.showerror("Error", "No image selected")

    button = tk.Button(root, text='Select Image', command=on_select_image, height=1, width=12, fg='black', font=('JetBrains', 10, 'bold'))
    button.pack(pady=10)


    space = tk.Label(root, text='', bg='SaddleBrown')
    space.pack()

    label = tk.Label(root, text='Select the filter:', font=('JetBrains Mono',16,), bg='SaddleBrown', fg='white')
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

    button = tk.Button(root, text='Process Image', command=on_process_image,height=1, width=25, fg='black', font=('JetBrains', 14, 'bold'))
    button.pack(pady=30)



def main():
    root.geometry('500x500')
    root.title('Image Denoising')
    root.configure(bg= 'SaddleBrown')

    interface()

    root.mainloop()

    

if __name__ == '__main__':
    main()