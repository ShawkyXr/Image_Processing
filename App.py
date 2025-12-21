import customtkinter as ctk
import os

def open_image_classification_app():
    os.system('python3 Image_Classification_App/main.py')

root = ctk.CTk()
root.geometry("600x400")
root.title('Image Processing App')

header = ctk.CTkLabel(
            root,
            text="Image Classification App",
            font=("JetBrains Mono", 30, "bold")
        )

header.pack(side=ctk.TOP, fill=ctk.X, pady=80)

content = ctk.CTkFrame(root, fg_color="transparent")
content.pack(fill=ctk.BOTH, expand=True)

button_frame = ctk.CTkFrame(content, fg_color="transparent")
button_frame.pack(pady=20)

bt_select = ctk.CTkButton(
            button_frame,
            text="Start",
            width=180,
            height=45,
            font=("JetBrains Mono", 13, "bold"),
            corner_radius=22,
            command=open_image_classification_app
        )
bt_select.pack()

if __name__ == "__main__":
    root.mainloop()