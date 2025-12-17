from tkinter import Tk, Frame, Button, Label, filedialog
from PIL import Image, ImageTk
import Prediction  

categories = ['cats', 'dogs']


class ImageClassificationApp:
    def __init__(self, root):
        self.root = root
        self.root.geometry("600x600")
        self.root.title('Image Classification')
        self.root.configure(bg='SaddleBrown')

        self.header = Label(
            self.root,
            text='Image Classification App',
            font=('JetBrains Mono', 28, 'bold'),
            bg='SaddleBrown',
            fg='white'
        )
        self.header.pack(side='top', fill='x', pady=50)

        self.fr1 = Frame(self.root, width=250, height=250, bg='white')
        self.fr1.place(x=100, y=155)

        self.lbl2 = Label(self.root, bg='SaddleBrown', fg='white')
        self.lbl2.place(x=420, y=260)

        self.fr3 = Frame(self.root, width=110, height=30, bg='white')
        self.fr3.place(x=170, y=480)

        self.fr4 = Frame(self.root, width=75, height=30, bg='white')
        self.fr4.place(x=190, y=420)

        self.bt1 = Button(
            self.fr3,
            text='Select Image',
            bg='white',
            fg='black',
            font=('JetBrains', 10, 'bold'),
            command=self.select_image
        )
        self.bt1.place(x=0, y=0)

        self.bt2 = Button(
            self.fr4,
            text='Predict',
            bg='white',
            fg='black',
            font=('JetBrains', 10, 'bold'),
            command=self.predict_image
        )
        self.bt2.place(x=0, y=0)

        self.lbl1 = Label(
            self.root,
            text='Predictions',
            bg='SaddleBrown',
            fg='white',
            font=('JetBrains', 10, 'bold')
        )
        self.lbl1.place(x=450, y=225)

        self.image_label = Label(self.fr1, bg='white')
        self.image_label.place(x=0, y=0)

        self.selected_image_path = None

    def load_and_display_image(self, image_path):
        try:
            image = Image.open(image_path)
            image = image.resize((250, 250), Image.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            self.image_label.config(image=photo)
            self.image_label.image = photo
            self.selected_image_path = image_path
        except Exception as e:
            print(f"Error loading image: {e}")

    def select_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.load_and_display_image(file_path)

    def predict_image(self):
        if not self.selected_image_path:
            print("No image selected.")
            return

        try:
            predictions = Prediction.predict_image_all_models(
                self.selected_image_path
            )

            result_text = ""
            for model, pred in predictions.items():
                result_text += f"{model}: {pred}\n"

            self.lbl2.config(
                text=result_text,
                font=('JetBrains', 11, 'bold'),
                justify='left'
            )

        except Exception as e:
            print(f"Prediction error: {e}")


if __name__ == "__main__":
    root = Tk()
    app = ImageClassificationApp(root)
    root.mainloop()
