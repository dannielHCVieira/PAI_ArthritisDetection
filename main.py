from email.mime import image
import string
import tkinter as tk
from time import time
from tkinter import *
from tkinter import messagebox
from tkinter.filedialog import askopenfilename, asksaveasfilename, askdirectory
from tkinter.scrolledtext import ScrolledText

import cv2 as cv
from PIL import ImageTk, Image
from PIL.Image import Resampling
import threading

import Operations
from Operations import matchTemplatePrivate
import numpy as np

import sys

from classes.draw_and_crop import create_environment


class MenuBar(Menu):
    def __init__(self, ad):
        Menu.__init__(self, ad)
        imageOperations = ImageOperations()

        # Opções (Open, Exit)
        file = Menu(self, tearoff=False)
        file.add_command(label="Open", command=imageOperations.openImage)
        file.add_separator()
        file.add_command(label="Open Folder", command=imageOperations.openFolder)
        file.add_separator()
        file.add_command(label="Exit", underline=1, command=self.quit)
        self.add_cascade(label="File", underline=0, menu=file)

        # Opções do Tools (Cut/Select subregion, Search region on image)
        tools = Menu(self, tearoff=0)
        tools.add_command(label="Cut/Select", command=imageOperations.select_area)
        tools.add_command(label="Search...", command=imageOperations.match_template_cnn)

        # Menus de treinamento, teste e predição única
        train = Menu(self, tearoff=0)
        train.add_command(label="Train XGBoost", command=imageOperations.trainXGBoost)
        train.add_command(label="Train DeepLearning", command=imageOperations.trainDL)
        train.add_command(label="Train SVM", command=imageOperations.trainSVM)

        predict = Menu(self, tearoff=0)
        predict.add_command(label="Predict XGBoost", command=imageOperations.predictXGBoost)
        predict.add_command(label="Predict DeepLearning", command=imageOperations.predictDL)
        predict.add_command(label="Predict SVM", command=imageOperations.predictSVM)

        tools.add_cascade(label="Train", menu=train)
        tools.add_cascade(label="Predict", menu=predict)

        test = Menu(self, tearoff=0)
        test.add_command(label="Test XGBoost", command=imageOperations.testXGBoost)
        test.add_command(label="Test DeepLearning", command=imageOperations.testDL)
        test.add_command(label="Test SVM", command=imageOperations.testSVM)

        tools.add_cascade(label="Test", menu=test)

        self.add_cascade(label="Tools", menu=tools)

        # Opções do Help (About the program)
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
    #funções auxiliares para teste e predição
    def testDL(self):
        thread = threading.Thread(target=self.testDLThread)
        thread.start()

    def testDLThread(self):
        before = time()
        report = Operations.testDL(folder + "\\test_preprocessed")
        spent = time() - before
        print("Total time to test: " + str(spent) + " s")
        print("Testing metrics:")
        print(report)
        self.showCM('DL')
    
    def testXGBoost(self):
        thread = threading.Thread(target=self.testXGBoostThread)
        thread.start()

    def testXGBoostThread(self):
        before = time()
        report = Operations.testXGBoost(folder + "\\test_preprocessed")
        spent = time() - before
        print("Total time to test: " + str(spent) + " s")
        print("Testing metrics:")
        print(report)
        self.showCM('XGB')

    def testSVM(self):
        thread = threading.Thread(target=self.testSVMThread)
        thread.start()

    def testSVMThread(self):
        before = time()
        report = Operations.testSVM(folder + "\\test_preprocessed")
        spent = time() - before
        print("Total time to test: " + str(spent) + " s")
        print("Testing metrics:")
        print(report)
        self.showCM('SVM')

    #funções auxiliares para treinamento
    def trainDL(self):
        thread = threading.Thread(target=Operations.trainDL, args=(folder + "\\train_preprocessed", folder + "\\val_preprocessed"))
        thread.start()

    def trainXGBoost(self):
        thread = threading.Thread(target=Operations.trainXGBoost, args=(folder + "\\train_preprocessed",))
        thread.start()
        #Operations.trainXGBoost(folder + "\\train_preprocessed")

    def trainSVM(self):
        thread = threading.Thread(target=Operations.trainSVM, args=(folder + "\\train_preprocessed",))
        thread.start()

    # Carrega imagens para o programa
    def loadImage(self, path: string):
        img = Image.open(path)
        img_resized = img.resize((224, 224), Resampling.LANCZOS)
        self.displayImage(img_resized)

    # Mostra a especificação  
    def displayImage(self, img):
        print(type(img))
        create_environment(img)

    def showCM(self, type: string):
        img = cv.imread("results/"+ type.lower() +"_cm.png")
        cv.imshow(type, img)
        cv.waitKey(0)
        cv.destroyAllWindows()

    def predictXGBoost(self):
        img = Operations.apply_match_template(image)
        img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
        class_pred = Operations.predict("XG", img)
        self.classify_image(class_pred)


    def predictSVM(self):
        img = Operations.apply_match_template(image)
        img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
        class_pred = Operations.predict("SVM", img)
        self.classify_image(class_pred)

    def predictDL(self):
        img = Operations.apply_match_template(image)
        img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
        class_pred = Operations.predict("DL", img)
        self.classify_image(class_pred)

    def classify_image(self, class_pred: int):
    #classifica as imagens de forma binária e multi-classe
        if class_pred > 1:
            print("Osteoarthritis detected. Class " + str(class_pred))
        else:
            print("No osteoarthritis detected.", end="")
            if(class_pred == 1):
                print(" Doubtful joint space narrowing. Risk of evolving into Osteoarthritis.", end="")
            print(" Class " + str(class_pred))

    # Abre a imagem no canvas
    def openFolder(self):
        global folder
        folder = askdirectory()
        thread = threading.Thread(target=self.preprocess)
        thread.start()

    def preprocess(self):
        print(folder + "/train")
        Operations.preprocess_images(folder + "\\train")
        print(folder + "/test")
        Operations.preprocess_images(folder + "\\test")
        print(folder + "/val")
        Operations.preprocess_images(folder + "\\val")
        print("Finished!")

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
        # lasx, lasy = event.x, event.y

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
        # apaga possíveis desenhos antigos
        ad.image_area.delete("desenho")

        top_left, bottom_right = matchTemplatePrivate(cv.imread(filename, 0))

        # identifica area com um retângulo
        ad.image_area.create_rectangle((top_left[0], top_left[1], bottom_right[0], bottom_right[1]), outline="red",
                                       width=2, tags="desenho")


# Link https://stackoverflow.com/questions/68198575/how-can-i-displaymy-console-output-in-tkinter
class PrintLogger(object):  # create file like object
    def __init__(self, textbox):  # pass reference to text widget
        self.textbox = textbox  # keep ref

    def write(self, text):
        self.textbox.configure(state="normal")  # make field editable
        self.textbox.insert("end", text)  # write text to textbox
        self.textbox.see("end")  # scroll to end
        self.textbox.configure(state="disabled")  # make field readonly

    def flush(self):  # needed for file like object
        pass

class MainApp(Tk):
    def __init__(self):
        Tk.__init__(self)   
        self.root = Frame(self)
        self.root.pack(fill=BOTH, expand=YES)
        menubar = MenuBar(self)  # Declara menu bar

        self.config(menu=menubar)

        self.image_area = Canvas(self.root, width=224, height=224, bg="#C8C8C8")
        #self.image_area.place(x=80, y=50)

        self.log = ScrolledText(self, height=10, width=400, font=("consolas", "8", "normal"))
        
        self.image_area.pack(side=TOP, expand=YES)
        self.log.pack(side=BOTTOM, expand=NO)
    
        logger = PrintLogger(self.log)
        sys.stdout = logger
        sys.stderr = logger

if __name__ == "__main__":
    global ad, resultCNN
    resultCNN = "Saudável"
    ad = MainApp()
    ad.title('D.A.R.X.')
    ad.geometry('600x400')  # Resolução inicial
    ad.mainloop()

    # Diagnostico de Artrite por Raio X (DARX)
