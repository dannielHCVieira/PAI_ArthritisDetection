from email.mime import image
import string
import tkinter as tk
from tkinter import *
from tkinter import messagebox
from tkinter.filedialog import askopenfilename, asksaveasfilename
import cv2 as cv
from PIL import ImageTk, Image
from PIL.Image import Resampling

import numpy as np

from draw_and_crop import create_environment, crop_img

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
        tools.add_command(label="Cut/Select", command=imageOperations.select_area)
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

        self.image_area = Canvas(self, width=300, height=300, bg="#C8C8C8")
        self.image_area.place(x = 80, y = 50) #pack(side=tk.TOP, padx=(20,0), pady=(0,0))

        self.result = Label(self, bg="#4f4545", text=resultCNN )

class ImageOperations:

    def loadImage(self, path: string):
        img = Image.open(path)
        img_resized = img.resize((300, 300), Resampling.LANCZOS)
        self.displayImage(img_resized)
        
    def displayImage(self, img):
        print(type(img))
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
        global image, imageTK
        if filename:
            image = Image.open(filename)
            image = image.resize((300,300), Image.ANTIALIAS)
            imageTK = ImageTk.PhotoImage(image)
            ad.image_area.create_image(0,0, image=imageTK, anchor='nw')
        # return self.loadImage(filename)
        # root.deiconify()

    def get_xy(self, event):
        print("GETXY")
        ad.image_area.delete("desenho")
        global lasx, lasy
        lasx, lasy = event.x, event.y

    def draw_selection(self, event):
        print("DRAW_SELECTION")
        global refs_point
        ad.image_area.create_rectangle((lasx, lasy, event.x, event.y), width=2, tags="desenho")
        refs_point = [(lasx, lasy),(event.x,event.y)]
        #lasx, lasy = event.x, event.y


    def select_area(self):
        ad.image_area.bind("<Button-1>", self.get_xy)
        ad.image_area.bind("<ButtonRelease-1>", self.draw_selection)
        ad.image_area.bind("<Button-3>", self.crop_image)
    
    """ def handle_crop_image(self, event):
        print("Aopa")
        crop_img(image, ref_point=refs_point) """
    
    def crop_image(self, event):
        print("CROP")
        global refs_point
        if len(refs_point) == 2:
            img = self.imageTKtoCV2(image)
            crop_img = img[refs_point[0][1]:refs_point[1][1], refs_point[0][0]:
                                                                refs_point[1][0]]
            #cv.imshow("crop_img", crop_img)
            #cv.waitKey(0)

            path = asksaveasfilename(defaultextension=".png")
            print(path)
            cv.imwrite(path, crop_img)
    
    def imageTKtoCV2(self, image):
        pil_image = image.convert("RGB")
        open_cv_image = np.array(pil_image) 
        # Convert RGB to BGR 
        open_cv_image = open_cv_image[:, :, ::-1].copy()
        return open_cv_image



if __name__ == "__main__":
    global ad, resultCNN
    resultCNN = "Saudavél"
    ad=MainApp()
    ad.title('ArthritisDetec')
    ad.geometry('600x600') # Initial Resolution
    ad.mainloop()
