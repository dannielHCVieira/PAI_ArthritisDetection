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

from classes.draw_and_crop import create_environment

class MenuBar(Menu):
    def __init__(self, ad):
        Menu.__init__(self, ad)
        imageOperations = ImageOperations()

        # Opções (Open, Exit)
        file = Menu(self, tearoff=False)
        file.add_command(label="Open", command=imageOperations.openImage)
        file.add_separator()
        file.add_command(label="Exit", underline=1, command=self.quit)
        self.add_cascade(label="File", underline=0, menu=file)
        
        #Opções do Tools (Cut/Select subregion, Search region on image)
        tools = Menu(self, tearoff=0)  
        tools.add_command(label="Cut/Select", command=imageOperations.select_area)
        tools.add_command(label="Search...", command=imageOperations.match_template_cnn)
        self.add_cascade(label="Tools", menu=tools) 

        #Opções do Help (About the program)
        help = Menu(self, tearoff=0)  
        help.add_command(label="About", command=self.about)  
        self.add_cascade(label="Help", menu=help)

    def exit(self):
        self.exit()

    def about(self):
            messagebox.showinfo(title='ArthritisDetec', 
            message='''Software made for the class of Processing and Analysis of Images in PUC Minas.
            
            By Danniel Vieira, João Amorim and Maria Fernanda''')

class ImageOperations:

    # Carrega imagens para o programa
    def loadImage(self, path: string):
        img = Image.open(path)
        img_resized = img.resize((224, 224), Resampling.LANCZOS)
        self.displayImage(img_resized)
        
    # Mostra a especificação  
    def displayImage(self, img):
        print(type(img))
        create_environment(img)

    # Abre a imagem no canvas 
    def openImage(self):
        global image, imageTK, filename

        filename = askopenfilename()
        if filename:
            image = Image.open(filename)
            imageTK = ImageTk.PhotoImage(image)
            ad.image_area.create_image(0, 0, image=imageTK, anchor='nw')

    # Obter a posição x,y do ponteiro do mouse
    def get_xy(self, event):
        ad.image_area.delete("desenho")
        global lasx, lasy
        lasx, lasy = event.x, event.y

    # Cria um retângulo com baseado na última e atual posição do ponteiro do mouse
    def draw_selection(self, event):
        global refs_point
        ad.image_area.create_rectangle((lasx, lasy, event.x, event.y), width=2, tags="desenho", outline="red")
        refs_point = [(lasx, lasy), (event.x, event.y)]
        #lasx, lasy = event.x, event.y

    # Vincule os botões para selecionar uma área para cortar
    def select_area(self):
        ad.image_area.delete("desenho")
        ad.image_area.bind("<Button-1>", self.get_xy)
        ad.image_area.bind("<ButtonRelease-1>", self.draw_selection)
        ad.image_area.bind("<Button-3>", self.crop_image)

    # Corta e salva a imagem
    def crop_image(self, event):
        global refs_point
        if len(refs_point) == 2:
            img = self.imageTKtoCV2(image)
            crop_img = img[refs_point[0][1]:refs_point[1][1], refs_point[0][0]:
                                                                refs_point[1][0]]
            path = asksaveasfilename(defaultextension=".png")
            print(path)
            # se um caminho valido foi selecioado salvamos a imagem
            if path != "":
                cv.imwrite(path, crop_img)

            ad.image_area.delete("desenho")
    
    # Converte um objeto imagem em uma imagem cv2(matriz)
    def imageTKtoCV2(self, image):
        pil_image = image.convert("RGB")
        open_cv_image = np.array(pil_image) 
        open_cv_image = open_cv_image[:, :, ::-1].copy()
        return open_cv_image

    def match_template_cnn(self):
        #apaga possíveis desenhos antigos
        ad.image_area.delete("desenho")
        METHOD = cv.TM_CCOEFF

        #lê novamente a imagem para evitar dados quebrados
        img = cv.imread(filename, 0)
        edged_img = cv.Canny(img, 30, 200)
        img2 = img.copy()

        #carrega template para joelho esquerdo e direito
        template_l = cv.imread("templates/template_L.png", 0)
        template_r = cv.imread("templates/template_R.png", 0)

        #encontra contornos
        edged_template_l = cv.Canny(template_l, 30, 200)
        edged_template_r = cv.Canny(template_r, 30, 200)

        w_l, h_l = template_l.shape[::-1]
        w_r, h_r = template_l.shape[::-1]

        # aplica o math template em ambas as imagens de template
        res_l = cv.matchTemplate(edged_img, edged_template_l, METHOD)
        res_r = cv.matchTemplate(edged_img, edged_template_r, METHOD)

        min_val_l, max_val_l, min_loc_l, max_loc_l = cv.minMaxLoc(res_l)
        min_val_r, max_val_r, min_loc_r, max_loc_r = cv.minMaxLoc(res_r)

        #define qual imagem deu melhor match
        if max_val_r > max_val_l:
            top_left = max_loc_r
            bottom_right = (top_left[0] + w_r, top_left[1] + h_r)
        else:
            top_left = max_loc_l
            bottom_right = (top_left[0] + w_l, top_left[1] + h_l)

        #identifica area com um retângulo
        ad.image_area.create_rectangle((top_left[0], top_left[1], bottom_right[0], bottom_right[1]), outline="red", width=2, tags="desenho")


class MainApp(Tk):
    def __init__(self):
        Tk.__init__(self)
        menubar = MenuBar(self) # Declara menu bar
        
        self.config(menu=menubar)

        self.image_area = Canvas(self, width=224, height=224, bg="#C8C8C8")
        self.image_area.place(x = 80, y = 50) 

        self.result = Label(self, bg="#4f4545", text=resultCNN )

if __name__ == "__main__":
    global ad, resultCNN
    resultCNN = "Saudável"
    ad=MainApp()
    ad.title('ArthritisDetec')
    ad.geometry('400x400') # Resolução inicial
    ad.mainloop()
