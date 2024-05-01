import tkinter as tk

import cv2
from PIL import Image, ImageTk
import model


class CameraApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        # Open the camera
        self.cap = cv2.VideoCapture(0)

        # Create a canvas that can fit the above video source size
        self.canvas = tk.Canvas(window, width=self.cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                                height=self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas.pack()

        # Add a text box
        self.textbox = tk.Entry(window)
        self.textbox.pack()

        self.capture_button = tk.Button(window, text="Clear", command=self.clearText)
        self.capture_button.pack()

        # After the window is displayed, set up the video feed
        self.delay = 10
        self.update()

        self.window.mainloop()

    def clearText(self):
        self.textbox.delete(0, tk.END)

    def update(self):
        ret, frame = self.cap.read()
        if ret:
            # Perform text recognition
            text = self.recognize_text(frame)

            # Update the text box with recognized text
            self.textbox.delete(0, tk.END)
            self.textbox.insert(0, text)

            # Display the frame in the GUI
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        self.window.after(self.delay, self.update)

    def recognize_text(self, frame):
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        predict = model.predict(frame)

        # Apply image preprocessing if needed (e.g., thresholding, noise removal)
        # You may need to adjust these parameters depending on your use case
        # For example:
        # gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        # Use Tesseract to perform OCR (Optical Character Recognition)
        # Make sure you have Tesseract installed and configured properly on your system
        # text = pytesseract.image_to_string(gray)
        text = self.textbox.get() + predict
        return text


# Create a window and pass it to the CameraApp class
window = tk.Tk()
app = CameraApp(window, "Live Camera with Text Recognition")
