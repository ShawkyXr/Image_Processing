from tkinter import Tk, Frame, Button, Label, filedialog
from PIL import Image, ImageTk
import Prediction  # Import the Prediction module

categories = ['cats','dogs']


class ImageClassificationApp:
    def __init__(self, root):
        self.root = root
        self.root.geometry("600x600")
        self.root.title('Image Classification')
        self.root.configure(bg='SaddleBrown')

        self.header = Label(self.root, text='Image Classification App', font=('JetBrains Mono', 28, 'bold'), bg='SaddleBrown', fg='white')
        self.header.pack(side='top', fill='x', pady=50)
        
        self.fr1 = Frame(self.root, width='250', height='250', bg='white')
        self.fr1.place(x=100, y=155)

        self.fr2 = Frame(self.root, width='100', height='30', bg='white')
        self.fr2.place(x=450, y=250)
        
        self.lbl2 = Label(self.root)
        self.lbl2.place(x=465, y=250)


        self.fr3 = Frame(self.root, width='110', height='30', bg='white')
        self.fr3.place(x=170, y=480)

        self.fr4 = Frame(self.root, width='75', height='30', bg='white')
        self.fr4.place(x=190, y=420)

        self.bt1 = Button(self.fr3, text='Select Image', bg='white', fg='black', font=('JetBrains', 10, 'bold'), command=self.select_image)
        self.bt1.place(x=0, y=0)

        self.bt2 = Button(self.fr4, text='Predict', bg='white', fg='black', font=('JetBrains', 10, 'bold'), command=self.predict_image)
        self.bt2.place(x=0, y=0)

        self.lbl1 = Label(self.root, text='Prediction', bg='SaddleBrown', fg='white', font=('JetBrains', 10, 'bold'))
        self.lbl1.place(x=460, y=225)

        self.image_label = Label(self.fr1, bg='white')
        self.image_label.place(x=0, y=0)

        self.selected_image_path = None  # Store the path of the selected image 
        # imagepath
    def load_and_display_image(self, image_path):
        try:
            image = Image.open(image_path)
            image = image.resize((250, 250), Image.LANCZOS) # Resize the image to fit the display area
            photo = ImageTk.PhotoImage(image)
            self.image_label.config(image=photo)
            self.image_label.image = photo  # Keep a reference to the image to prevent it from being garbage collected
            self.selected_image_path = image_path  # Store the path of the selected image
        except Exception as e:
            print(f"Error loading image: {e}")

    def select_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.load_and_display_image(file_path)

    def predict_image(self):
        if self.selected_image_path:
            try:
                # Call the prediction function from the Prediction module
                prediction = Prediction.predict_image(self.selected_image_path , categories)
                self.lbl2.config(text=f' {prediction}', font=('JetBrains', 13, 'bold') )
            except Exception as e:
                print(f"Error predicting image: {e}")
        else:
            print("No image selected for prediction.")

if __name__ == "__main__":
    root = Tk()
    app = ImageClassificationApp(root)
    root.mainloop()