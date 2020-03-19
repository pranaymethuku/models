#givememyredline
from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image
import image_detection_gui
import video_detection_gui
import os

root = Tk()
root.title('Tiered Object Recognition')
OPTIONS = [
"Tier 1",
"Tier 2",
"Tier 3",
"Tier 4"
]

VIDEOS = [".mov", ".mp4", ".flv", ".avi", ".ogg", ".wmv"]

variable = StringVar(root)
variable.set(OPTIONS[0]) # default value

option = OptionMenu(root, variable, *OPTIONS)
option.pack()

def open():
    global my_image
    root.filename = filedialog.askopenfilename(initialdir="", title="Select A File", filetypes=(("jpg files", "*.jpg"),("all files", "*.*")))
    file_extension = os.path.splitext(root.filename)
    if file_extension[1] == ".jpg":
        image_detection_gui.main(root.filename, variable.get())
        my_image = ImageTk.PhotoImage(Image.open("predicted.jpg"))
        my_image_label = Label(image=my_image).pack()

    if file_extension[1] in VIDEOS:
        video_detection_gui.main()

my_btn = Button(root, text="Open File", command=open).pack()

root.mainloop()