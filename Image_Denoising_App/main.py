import cv2
import matplotlib.pyplot as plt
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from Gaussian_Filter import denoise_image as gaussian_filter
from Median_Filter import denoise_image as median_filter
from Average_Filter import denoise_image as average_filter
import customtkinter as ctk

# Initialize main window
root = ctk.CTk()
root.geometry("900x800")
root.title('Image Processing App')

# Available filters
FILTERS = {
    "Gaussian": gaussian_filter,
    "Median": median_filter,
    "Average": average_filter
}

# Function to show images using matplotlib
def show_image(original_img, denoised_img):
    plt.figure(figsize=(8,5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(original_img)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Denoised Image")
    plt.imshow(denoised_img)
    plt.axis("off")

    plt.show()

# Function to display image in GUI
def display_image(img_array):
    img = Image.fromarray(img_array)
    img = img.resize((300, 300))
    tk_img = ImageTk.PhotoImage(img)

    if hasattr(root, "img_label"):
        root.img_label.configure(image=tk_img)
        root.img_label.image = tk_img
    else:
        root.img_label = ctk.CTkLabel(root, image=tk_img)
        root.img_label.image = tk_img
        root.img_label.pack(pady=10)

# Interface
def interface():
    path_var = ctk.StringVar(value="")
    filter_var = ctk.StringVar(value="Gaussian")

    # Heading
    ctk.CTkLabel(root, text='Image Denoising App', font=('JetBrains Mono', 28, 'bold')).pack(side=ctk.TOP, fill=ctk.X, pady=20)
    ctk.CTkLabel(root, text='Select image from your files', font=('JetBrains Mono',18,'bold')).pack(pady=10)

    # Select Image Button
    def on_select_image():
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        if file_path:
            path_var.set(file_path)
            messagebox.showinfo("Success", "Image selected successfully")
            img = cv2.imread(file_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            display_image(img)
        else:
            messagebox.showerror("Error", "No image selected")

    ctk.CTkButton(
        root,
        text='Select Image',
        command=on_select_image,
        height=40,
        width=200,
        font=('JetBrains Mono', 16, 'bold')
    ).pack(pady=20)

    # Filter Selection
    ctk.CTkLabel(root, text='Select the filter:', font=('JetBrains Mono',16,'bold')).pack(pady=10)
    filter_options = list(FILTERS.keys())
    filter_menu = ctk.CTkOptionMenu(root, variable=filter_var, values=filter_options)
    filter_menu.pack(pady=10)

    # Denoise Button
    def on_process_image():
        if not path_var.get():
            messagebox.showerror("Error", "No image selected")
            return
        original_img = cv2.imread(path_var.get())
        if original_img is None:
            messagebox.showerror("Error", "Failed to load image")
            return
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        denoised_img = FILTERS[filter_var.get()](original_img)
        show_image(original_img, denoised_img)

    ctk.CTkButton(
        root,
        text='Denoise Image',
        command=on_process_image,
        height=40,
        width=200,
        font=('JetBrains Mono', 16, 'bold')
    ).pack(pady=30)

def main():
    interface()
    root.mainloop()

if __name__ == '__main__':
    main()
