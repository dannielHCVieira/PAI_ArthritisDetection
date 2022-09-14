import string
import tkinter
from tkinter import *
from tkinter import messagebox
from tkinter.filedialog import askopenfilename
import cv2 as cv
from PIL import ImageTk, Image
from PIL.Image import Resampling

from draw_and_crop import create_environment


class MenuBar(Menu):
    def __init__(self, ad):
        Menu.__init__(self, ad)
        imageOperations = ImageOperations()

        #File options (Open, Exit)
        file = Menu(self, tearoff=False)
        file.add_command(label="Open", command=imageOperations.openImage)
        file.add_separator()
        file.add_command(label="Exit", underline=1, command=self.quit)
        self.add_cascade(label="File", underline=0, menu=file)
        
        #Tools options (Cut/Select subregion, Search region on image)
        tools = Menu(self, tearoff=0)  
        tools.add_command(label="Cut/Select")
        tools.add_command(label="Search...")  
        self.add_cascade(label="Tools", menu=tools) 

        #Help options (About the program)
        help = Menu(self, tearoff=0)  
        help.add_command(label="About", command=self.about)  
        self.add_cascade(label="Help", menu=help)

        #Open image

    def exit(self):
        self.exit()

    def about(self):
            messagebox.showinfo(title='ArthritisDetec', 
            message='''Software made for the class of Processing and Analysis of Images in PUC Minas.
            
            By Danniel Vieira, João Amorim and Maria Fernanda''')


class MainApp(Tk):
    def __init__(self):
        Tk.__init__(self)
        menubar = MenuBar(self) # Declaring menu bar
        self.config(menu=menubar)

class ImageOperations:

    def loadImage(self, path: string):
        img = Image.open(path)
        img_resized = img.resize((300, 300), Resampling.LANCZOS)
        self.displayImage(img_resized)
    def displayImage(self, img):
        create_environment(img)
        # test = ImageTk.PhotoImage(img)
        #
        # label1 = tkinter.Label(image=test)
        # label1.image = test
        # w = label1.winfo_width()
        # h = label1.winfo_height()
        #
        # label1.pack(anchor="center")#place(x=300, y=300, anchor="center")

    def openImage(self):
        # root.withdraw()
        filename = askopenfilename()
        return self.loadImage(filename)
        # root.deiconify()

if __name__ == "__main__":
    ad=MainApp()
    ad.title('ArthritisDetec')
    ad.geometry('600x600') # Initial Resolution
    ad.mainloop()
