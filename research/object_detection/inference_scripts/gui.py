#givememyredline
import tkinter as tk
from tkinter import filedialog, messagebox
import threading
from PIL import ImageTk, Image
import image_detection_gui
import video_detection_gui
import os
import imageio

root = tk.Tk()
root.title('Tiered Object Recognition')


VIDEOS = [".mov", ".mp4", ".flv", ".avi", ".ogg", ".wmv"]

def set_options(*args):
    """
    Function to configure options for second drop down
    """
    global option, option2, menu2
    a = ['model1', 'model2', 'model3']
    b = ['model4', 'model5', 'model6']
    c = ['model7', 'model8', 'model9']
    d = ['model10', 'model11', 'model12']

    # check something has been selected
    if option.get() == '(select)':
        return None

    # refresh option menu
    option2.set('(select)')
    menu2['menu'].delete(0, 'end')

    # pick new set of options
    if option.get() == 'Tier 1':
        new_options = a
    elif option.get() == 'Tier 2':
        new_options = b
    elif option.get() == 'Tier 3':
        new_options = c
    elif option.get() == 'Tier 4':
        new_options = d

    # add new options in
    for item in new_options:
        menu2['menu'].add_command(label=item, command=tk._setit(option2, item))


def open():
    global my_image
    root.filename = filedialog.askopenfilename(initialdir="", title="Select A File", filetypes=(("jpg files", "*.jpg"),("all files", "*.*")))
    file_extension = os.path.splitext(root.filename)
    if option.get() == '(select)' or option2.get() == '(select)':
        messagebox.showerror("Error","You must select a Tier and Model")
    else:
        if file_extension[1] == ".jpg":
            image_detection_gui.main(root.filename, option.get())
            my_image = ImageTk.PhotoImage(Image.open("predicted.jpg"))
            my_image_label = tk.Label(image=my_image).pack()

        if file_extension[1] in VIDEOS:
            my_label = tk.Label(root)
            my_label.pack()
            video = imageio.get_reader(root.filename)
            thread = threading.Thread(target=stream, args=(my_label, video))
            thread.daemon = 1
            thread.start()


def stream(label, video):
    for image in video.iter_data():
        frame_image = ImageTk.PhotoImage(Image.fromarray(image))
        label.config(image=frame_image)
        label.image = frame_image

# drop down to determine second drop down
option = tk.StringVar(root)
option.set('(select)')

menu1 = tk.OptionMenu(root, option, 'Tier 1', 'Tier 2', 'Tier 3', 'Tier 4')
menu1.pack()

# trace variable and change second drop down
option.trace('w', set_options)

# second drop down
option2 = tk.StringVar(root)
option2.set('(select)')

menu2 = tk.OptionMenu(root, option2, '(select)')
menu2.pack()
set_options()

my_btn = tk.Button(root, text="Open File", command=open).pack()

root.mainloop()