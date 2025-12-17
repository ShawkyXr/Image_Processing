import customtkinter as ctk
from tkinter import filedialog
from PIL import Image
import Prediction

categories = ['cats', 'dogs']


class ImageClassificationApp:
    def __init__(self, root):
        self.root = root
        self.root.geometry("1000x900")
        self.root.title("Image Classification")

        # ---------------- Grid Config ----------------
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        # ---------------- Header ----------------
        self.header = ctk.CTkLabel(
            self.root,
            text="Image Classification App",
            font=("JetBrains Mono", 30, "bold")
        )
        self.header.grid(row=0, column=0, pady=30)

        # ---------------- Content ----------------
        self.content = ctk.CTkFrame(self.root, fg_color="transparent")
        self.content.grid(row=1, column=0, sticky="n")

        # ---------------- Image Frame ----------------
        self.fr1 = ctk.CTkFrame(
            self.content,
            width=250,
            height=250,
            corner_radius=15
        )
        self.fr1.grid(row=0, column=0, columnspan=2, pady=20)

        self.image_label = ctk.CTkLabel(self.fr1, text="")
        self.image_label.pack()

        # ---------------- Prediction Title ----------------
        self.lbl1 = ctk.CTkLabel(
            self.content,
            text="Model Comparison",
            font=("JetBrains Mono", 16, "bold")
        )
        self.lbl1.grid(row=1, column=0, columnspan=2, pady=(20, 10))

        # ---------------- Table Frame ----------------
        self.table_frame = ctk.CTkFrame(self.content, corner_radius=12)
        self.table_frame.grid(row=2, column=0, columnspan=2, padx=20, pady=10)

        headers = ["Model", "Prediction", "Accuracy", "Precision", "Recall", "F1-score"]
        for col, text in enumerate(headers):
            ctk.CTkLabel(
                self.table_frame,
                text=text,
                font=("JetBrains Mono", 14, "bold")
            ).grid(row=0, column=col, padx=15, pady=10)

        # ---------------- Buttons (CENTERED UNDER TABLE) ----------------
        self.button_frame = ctk.CTkFrame(self.content, fg_color="transparent")
        self.button_frame.grid(row=3, column=0, columnspan=2, pady=40)

        self.bt_select = ctk.CTkButton(
            self.button_frame,
            text="Select Image",
            width=180,
            height=45,
            font=("JetBrains Mono", 13, "bold"),
            corner_radius=22,
            command=self.select_image
        )
        self.bt_select.grid(row=0, column=0, padx=20)

        self.bt_predict = ctk.CTkButton(
            self.button_frame,
            text="Predict",
            width=140,
            height=45,
            font=("JetBrains Mono", 13, "bold"),
            corner_radius=22,
            fg_color="#8B5A2B",
            hover_color="#A0522D",
            command=self.predict_image
        )
        self.bt_predict.grid(row=0, column=1, padx=20)

        self.selected_image_path = None

    # ---------------- Image Handling ----------------
    def load_and_display_image(self, image_path):
        try:
            image = Image.open(image_path)

            ctk_image = ctk.CTkImage(
                light_image=image,
                dark_image=image,
                size=(250, 250)
            )

            self.image_label.configure(image=ctk_image)
            self.image_label.image = ctk_image
            self.selected_image_path = image_path

        except Exception as e:
            print(f"Error loading image: {e}")

    def select_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.load_and_display_image(file_path)

    # ---------------- Prediction & Table Fill ----------------
    def predict_image(self):
        if not self.selected_image_path:
            print("No image selected.")
            return

        try:
            results = Prediction.predict_image_all_models(
                self.selected_image_path
            )

            # Clear previous rows
            for widget in self.table_frame.winfo_children():
                if int(widget.grid_info()["row"]) > 0:
                    widget.destroy()

            for row, (model, data) in enumerate(results.items(), start=1):

                ctk.CTkLabel(
                    self.table_frame,
                    text=model,
                    font=("JetBrains Mono", 13)
                ).grid(row=row, column=0, padx=10, pady=6)

                ctk.CTkLabel(
                    self.table_frame,
                    text=data["prediction"].upper(),
                    font=("JetBrains Mono", 13, "bold"),
                    text_color="lightgreen"
                ).grid(row=row, column=1)

                ctk.CTkLabel(
                    self.table_frame,
                    text=f"{data['accuracy']*100:.1f}%"
                ).grid(row=row, column=2)

                ctk.CTkLabel(
                    self.table_frame,
                    text=f"{data['precision']:.2f}"
                ).grid(row=row, column=3)

                ctk.CTkLabel(
                    self.table_frame,
                    text=f"{data['recall']:.2f}"
                ).grid(row=row, column=4)

                ctk.CTkLabel(
                    self.table_frame,
                    text=f"{data['f1']:.2f}"
                ).grid(row=row, column=5)

        except Exception as e:
            print(f"Prediction error: {e}")


if __name__ == "__main__":
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("dark-blue")

    root = ctk.CTk()
    app = ImageClassificationApp(root)
    root.mainloop()
