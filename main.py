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
        self.capture_button = tk.Button(window, text="Capture", command=self.capture)
        self.capture_button.pack()

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
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        self.window.after(self.delay, self.update)

    def capture(self):
        ret, frame = self.cap.read()
        if ret:
            # Perform text recognition
            text = self.recognize_text(frame)

            # Update the text box with recognized text
            self.textbox.delete(0, tk.END)
            self.textbox.insert(0, text)

    def recognize_text(self, frame):
        # Convert the frame to grayscale
        cv2.imwrite("./caputre.jpeg", frame)
        predict = model.predict("./caputre.jpeg")

        text = self.textbox.get() + predict
        return text


# Create a window and pass it to the CameraApp class
window = tk.Tk()
app = CameraApp(window, "Live Camera with Text Recognition")
