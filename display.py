import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import pytesseract

import model


# Function to upload image
def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        # Open the image file
        image = Image.open(file_path)
        # Display the image
        display_image(image)
        # Perform OCR
        extract_text(file_path)

# Function to display image
def display_image(image):
    # Resize image if necessary
    if image.width > 400 or image.height > 400:
        image.thumbnail((400, 400))
    # Convert image to Tkinter format
    photo = ImageTk.PhotoImage(image)
    # Display image on the label
    label.config(image=photo)
    label.image = photo

# Function to perform OCR
def extract_text(image):
    #text = pytesseract.image_to_string(image)
    text = model.predict(image)
    text_box.delete('1.0', tk.END)
    text_box.insert(tk.END, text)

# Main Tkinter window
root = tk.Tk()
root.title("Image to Text")

# Button to upload image
upload_button = tk.Button(root, text="Upload Image", command=upload_image)
upload_button.pack(pady=10)

# Label to display image
label = tk.Label(root)
label.pack()

# Text box to display extracted text
text_box = tk.Text(root, height=10, width=50)
text_box.pack(pady=10)

root.mainloop()
