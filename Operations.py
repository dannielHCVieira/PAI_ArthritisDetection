import os

import cv2 as cv
import matplotlib
import numpy as np
from imutils import paths
from keras import Sequential
from keras.layers import Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.utils import load_img, to_categorical
from keras_preprocessing.image import img_to_array, array_to_img, ImageDataGenerator
import tensorflow as tf
from matplotlib import pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix
from time import time
import pickle
from imutils import paths
from imutils import perspective
from imutils import contours
import imutils
import math
from scipy.spatial import distance as dist

import matplotlib.pyplot as plt
import tensorflow as tf
from imutils import paths
import os
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout,Flatten,Dense,Input
from keras import Sequential
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from scipy.constants import lb
import numpy as np
import cv2 as cv
import math
import imutils
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import confusion_matrix
import pandas as pd
from time import time


SVM = pickle.load(open('models/SVM.h5','rb'))  
XG = pickle.load(open('models/XGBoost.h5','rb')) 
DL = tf.keras.models.load_model('models/trained_model_mobileNet.h5')

def loadModel():
    SVM = pickle.load(open('models/SVM.h5','rb'))  #tf.keras.models.load_model('models/SVM.h5')
    XG = pickle.load(open('models/XGBoost.h5','rb'))  # tf.keras.models.load_model('models/XGBoost.h5')
    DL = tf.keras.models.load_model('models/trained_model_mobileNet.h5')

def matchTemplatePrivate(img):
    METHOD = cv.TM_CCOEFF

    # lê novamente a imagem para evitar dados quebrados
    edged_img = cv.adaptiveThreshold(img, 255,
                                     cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 21, 10)

    img2 = img.copy()

    # carrega template para joelho esquerdo e direito
    template_l = cv.imread("templates\\template_L.png", 0)
    template_r = cv.imread("templates\\template_R.png", 0)
    # encontra contornos
    edged_template_l = cv.adaptiveThreshold(template_l, 255,
                                            cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 21, 10)

    edged_template_r = cv.adaptiveThreshold(template_r, 255,
                                            cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 21, 10)

    w_l, h_l = template_l.shape[::-1]
    w_r, h_r = template_l.shape[::-1]

    # aplica o math template em ambas as imagens de template
    res_l = cv.matchTemplate(edged_img, edged_template_l, METHOD)
    res_r = cv.matchTemplate(edged_img, edged_template_r, METHOD)

    min_val_l, max_val_l, min_loc_l, max_loc_l = cv.minMaxLoc(res_l)
    min_val_r, max_val_r, min_loc_r, max_loc_r = cv.minMaxLoc(res_r)

    # define qual imagem deu melhor match
    if max_val_r > max_val_l:
        top_left = max_loc_r
        bottom_right = (top_left[0] + w_r, top_left[1] + h_r)
    else:
        top_left = max_loc_l
        bottom_right = (top_left[0] + w_l, top_left[1] + h_l)

    return top_left, bottom_right

def apply_match_template(image):
    image = img_to_array(image, dtype='uint8')

    x, y = matchTemplatePrivate(image)
    image = image[x[1]:y[1], x[0]:y[0]]
    image = array_to_img(image)
    image = image.resize((224, 224))
    # image.show()

    image = img_to_array(image, dtype='uint8')
    return image  # cv.cvtColor(image,cv.COLOR_GRAY2RGB) lembrar de convertar para RGB se necessário

def preprocess_images(dataset_path):
    preprocessed_path = dataset_path + "_preprocessed"

    if not os.path.isdir(preprocessed_path):
        os.mkdir(preprocessed_path)
        for folder in os.listdir(dataset_path):
            preprocess_path_sub = preprocessed_path + "\\" + folder
            for file in os.listdir(os.path.join(dataset_path + "\\" + folder)):
                # equaliza e flipa horizontalmente
                img = cv.imread(os.path.join(dataset_path + '\\' + folder + "\\" + file), 0)
                equ = cv.equalizeHist(img)
                flipped = cv.flip(equ, 1)
                flip_img = cv.flip(img, 1)

                # salva esse arquivo na pasta de pre-processados
                if not os.path.isdir(preprocess_path_sub):
                    os.mkdir(preprocess_path_sub)

                filename, file_type = file.split(".")
                filename_eq_path = preprocess_path_sub + "\\" + filename + "_equ." + file_type
                filename_eq_flip_path = preprocess_path_sub + "\\" + filename + "_flipped." + file_type
                filename_flip_path = preprocess_path_sub + "\\" + filename + "_flipped_only." + file_type

                cv.imwrite(preprocess_path_sub + "\\" + file, img)
                cv.imwrite(filename_eq_path, equ)
                cv.imwrite(filename_eq_flip_path, flipped)
                cv.imwrite(filename_flip_path, flip_img)

def find_distance_between_bones(img):
    gray = cv.GaussianBlur(img, (7, 7), 0)
    # perform edge detection, then perform a dilation + erosion to
    # close gaps in between object edges
    edged = cv.Canny(gray, 50, 100)
    edged = cv.dilate(edged, None, iterations=1)
    edged = cv.erode(edged, None, iterations=1)
    # find contours in the edge map
    cnts = cv.findContours(edged.copy(), cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    (cnts, _) = contours.sort_contours(cnts, method="top-to-bottom")
    colors = ((0, 0, 255), (240, 0, 159), (0, 165, 255), (255, 255, 0),
        (255, 0, 255))
    refObj = None
    
    menorDistancia = math.inf
    distancias = 0
    cont = 0
    
    for c in cnts:
        # if the contour is not sufficiently large, ignore it
        #if cv.contourArea(c) < 10:
        #   continue
        # compute the rotated bounding box of the contour
        box = cv.minAreaRect(c)
        box = cv.cv.BoxPoints(box) if imutils.is_cv2() else cv.boxPoints(box)
        box = np.array(box, dtype="int")
        # order the points in the contour such that they appear
        # in top-left, top-right, bottom-right, and bottom-left
        # order, then draw the outline of the rotated bounding
        # box
        box = perspective.order_points(box)
        # compute the center of the bounding box
        cX = np.average(box[:, 0])
        cY = np.average(box[:, 1])
        # if this is the first contour we are examining (i.e.,
        # the left-most contour), we presume this is the
        # reference object
        if refObj is None:
            # unpack the ordered bounding box, then compute the
            # midpoint between the top-left and top-right points,
            # followed by the midpoint between the top-right and
            # bottom-right
            (tl, tr, br, bl) = box
            (tlblX, tlblY) = midpoint(tl, bl)
            (trbrX, trbrY) = midpoint(tr, br)
            # compute the Euclidean distance between the midpoints,
            # then construct the reference object
            D = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
            refObj = (box, (cX, cY), D / 224)
            continue
        # draw the contours on the image
        orig = img.copy()
        cv.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
        cv.drawContours(orig, [refObj[0].astype("int")], -1, (0, 255, 0), 2)
        # stack the reference coordinates and the object coordinates
        # to include the object center
        refCoords = np.vstack([refObj[0], refObj[1]])
        objCoords = np.vstack([box, (cX, cY)])
        # loop over the original points
        for ((xA, yA), (xB, yB), color) in zip(refCoords, objCoords, colors):
            
            # draw circles corresponding to the current points and
            # connect them with a line
            cv.circle(orig, (int(xA), int(yA)), 5, color, -1)
            cv.circle(orig, (int(xB), int(yB)), 5, color, -1)
            cv.line(orig, (int(xA), int(yA)), (int(xB), int(yB)),
                color, 2)
            # compute the Euclidean distance between the coordinates,
            # and then convert the distance in pixels to distance in
            # units
            D = dist.euclidean((xA, yA), (xB, yB)) / refObj[2]

            distancias += D
            cont+=1
            if D< menorDistancia:
                menorDistancia = D

            (mX, mY) = midpoint((xA, yA), (xB, yB))
            cv.putText(orig, "{:.1f}in".format(D), (int(mX), int(mY - 10)),
                cv.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            # show the output image
            #cv.imshow("Image", orig)
            #cv.waitKey(0)
    
    if cont == 0:
        mediaDistancia = 0
    else:
        mediaDistancia = (distancias/cont)
        
    if menorDistancia == math.inf:
        menorDistancia = 0
    return mediaDistancia, menorDistancia

def midpoint(ptA, ptB):
	return (ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5

def count_black_pixels(image):
    # print(cropped)
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, bw = cv.threshold(image, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    # count non zero
    IMAGE_SIZE = 224 * 224
    response = cv.countNonZero(bw)
    return IMAGE_SIZE - response

def processImage(image):
    image = img_to_array(image, dtype='uint8')
    x, y = matchTemplatePrivate(image)
    image = image[x[1]:y[1], x[0]:y[0]]
    image = array_to_img(image)
    image = image.resize((224, 224))
    image = img_to_array(image)
    return cv.cvtColor(image, cv.COLOR_GRAY2RGB)

def testXGBoost(path_test):
    test_dataset = path_test

    test_images=list(paths.list_images(test_dataset))

    test_data=[]
    test_labels=[]

    print("Loading Images...", end="")
    for i in test_images:#adicionar nosso preprocessamento
        label=i.split(os.path.sep)[-2]
        test_labels.append(label)
        image = load_img(i,target_size=(224,224), color_mode="grayscale")
        image = processImage(image)
        test_data.append(image)
    print("OK!")


    print("Processing images...", end="")
    test_data=np.array(test_data, dtype='uint8')
    test_labels=np.array(test_labels)

    test_data_table = []
    for data in test_data:
        media, menor = find_distance_between_bones(data)
        count_black = count_black_pixels(data)
        test_data_table.append((media,menor,count_black))

    test_labels = np.array(test_labels, dtype=object)
    test_labels_int = test_labels.astype(np.dtype(np.int_))
    
    print("OK!")

    print("Loading model...",end="")
    xgb_model = pickle.load(open('models/XGBoost.h5','rb'))
    print("OK!")

    print("Testing model...",end="")
    xgb_predictions = xgb_model.predict(test_data_table)
    print("OK!")

    print("Generating Confusion Matrix...",end="")
    fig = plot_confusion_matrix(xgb_model, test_data_table, test_labels_int, cmap='Blues')
    plt.xlabel('', fontsize=18)
    plt.ylabel('', fontsize=18)
    plt.savefig("results\\xgb_cm.png")
    print("OK!")

    report = classification_report(test_labels_int, xgb_predictions, target_names=["0", "1", "2", "3", "4"])
    return report

def trainXGBoost(path_train):
    train_dataset = path_train

    train_images = list(paths.list_images(train_dataset))

    train_data = []
    train_labels = []

    print("Loading Images...", end="")
    for i in train_images:  # adicionar nosso preprocessamento
        label = i.split(os.path.sep)[-2]
        train_labels.append(label)
        image = load_img(i, target_size=(224, 224), color_mode="grayscale")
        image = processImage(image)
        train_data.append(image)

    print("OK!")

    print("Processing images..." ,end="")

    train_data = np.array(train_data, dtype='uint8')
    train_labels = np.array(train_labels)

    # ainda não acabou
    train_data_table = []
    for data in train_data:
        media, menor = find_distance_between_bones(data)
        count_black = count_black_pixels(data)
        train_data_table.append((media,menor,count_black))

    train_data_np = np.array(train_data_table, dtype=object)
    train_labels = np.array(train_labels, dtype=object)
    train_labels_int = train_labels.astype(np.dtype(np.int_))

    print("OK!")

    
    print("Training model...", end="")
    now = time()
    xgb_model = XGBClassifier(n_estimators = 650,
                      max_depth = 10,
                      learning_rate = 0.01,
                      subsample = 1,
                      random_state = 0
                     )
    now = time()
    xgb_history = xgb_model.fit(train_data_np, train_labels_int, verbose=False)
    total = time() - now
    print("OK!")


    xgb_filename = 'models/XGBoost.h5'
    pickle.dump(xgb_model, open(xgb_filename,'wb'))
    print("Model saved!")
    loadModel()

    print("Total time: " + total + "s")

def testSVM(path_test):
    test_dataset = path_test
    print("Loading Images...", end="")

    test_images=list(paths.list_images(test_dataset))

    test_data=[]
    test_labels=[]

    print("OK!")

    print("Processing images..." ,end="")
    for i in test_images:#adicionar nosso preprocessamento
        label=i.split(os.path.sep)[-2]
        test_labels.append(label)
        image = load_img(i,target_size=(224,224), color_mode="grayscale")
        image = processImage(image)
        test_data.append(image)
    
    test_data=np.array(test_data, dtype='uint8')
    test_labels=np.array(test_labels)

    test_data_table = []
    for data in test_data:
        media, menor = find_distance_between_bones(data)
        count_black = count_black_pixels(data)
        test_data_table.append((media,menor,count_black))
    print("OK!")

    print("Loading model...",end="")
    svm_model = pickle.load(open('models/SVM.h5','rb'))
    print("OK!")

    print("Testing model...",end="")
    svm_predict = svm_model.predict(test_data_table)
    print("OK!")

    print("Generating Confusion Matrix...",end="")

    cm = confusion_matrix(y_true=test_labels, y_pred=svm_predict)
    fig = plot_confusion_matrix(conf_mat=cm,
                                    figsize=(6, 6),
                                    class_names=["0", "1", "2", "3", "4"],
                                    # cmap='Greys',

                                    norm_colormap=matplotlib.colors.LogNorm())

    #fig = plot_confusion_matrix(svm_model, test_data_table, test_labels, cmap='Blues')
    plt.xlabel('', fontsize=18)
    plt.ylabel('', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.savefig("results\\svm_cm.png")
    print("OK!")

    report = classification_report(test_labels ,svm_predict, target_names=["0", "1", "2", "3", "4"])
    return report

def trainSVM(path_train):
    train_dataset = path_train
    print("Loading Images...", end="")
    train_images = list(paths.list_images(train_dataset))

    train_data = []
    train_labels = []

    for i in train_images:  # adicionar nosso preprocessamento
        label = i.split(os.path.sep)[-2]
        train_labels.append(label)
        image = load_img(i, target_size=(224, 224), color_mode="grayscale")
        image = processImage(image)
        train_data.append(image)
    print("OK!")

    print("Processing images..." ,end="")
    train_data = np.array(train_data, dtype='uint8')
    train_labels = np.array(train_labels)

    # ainda não acabou
    train_data_table = []
    for data in train_data:
        media, menor = find_distance_between_bones(data)
        count_black = count_black_pixels(data)
        train_data_table.append((media,menor,count_black))
    print("OK!")

    print("Training model...", end="")
    now = time()
    svm_model = SVC(kernel='rbf', class_weight='balanced').fit(train_data_table, train_labels)
    total = now - time()

    print("OK!")


    svm_filename = 'models/SVM.h5'
    pickle.dump(svm_model, open(svm_filename, 'wb'))
    loadModel()

    print("Model saved!")
    print("Total time: " + str(total) + "s")

def testDL(path_test):

    print("Loading Images...", end="")
    test_dataset = path_test

    test_images = list(paths.list_images(test_dataset))

    test_data = []
    test_labels = []
    print("OK!")

    print("Processing images..." ,end="")
    for i in test_images:  # adicionar nosso preprocessamento
        label = i.split(os.path.sep)[-2]
        test_labels.append(label)
        image = load_img(i, target_size=(224, 224), color_mode="grayscale")
        image = apply_match_template(image)
        image = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
        test_data.append(image)

    test_data = np.array(test_data, dtype='float32')
    test_labels = np.array(test_labels)

    test_labels = to_categorical(test_labels)

    print("OK!")

    BS = len(test_data)//10

    print("Loading model...",end="")

    aug = ImageDataGenerator()
    print(len(test_data))

    print("OK!")

    print("Testing model...",end="")

    predict = DL.predict(aug.flow(test_data), batch_size=BS)
    predict = np.argmax(predict, axis=1)
    print(predict)
    print(test_labels)

    print("OK!")
    report = classification_report(test_labels.argmax(axis=1), predict, target_names=["0", "1", "2", "3", "4"])

    print("Generating Confusion Matrix...",end="")
    cm = confusion_matrix(y_true=test_labels.argmax(axis=1), y_pred=predict)
    fig = plot_confusion_matrix(conf_mat=cm,
                                    figsize=(6, 6),
                                    class_names=["0", "1", "2", "3", "4"],
                                    # cmap='Greys',

                                    norm_colormap=matplotlib.colors.LogNorm())

    plt.xlabel('', fontsize=18)
    plt.ylabel('', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    # plt.show()
    plt.savefig("results\\dl_cm.png")
    print("OK!")

    return report

def trainDL(path_train, path_val):
    print("Loading Images...", end="")
    train_dataset = path_train
    val_dataset = path_val
    
    train_images = list(paths.list_images(train_dataset))
    val_images = list(paths.list_images(val_dataset))

    train_data = []
    train_labels = []

    val_data = []
    val_labels = []

    print("OK!")

    print("Processing images..." ,end="")

    for i in train_images:  # carrega imagens e preprocessa
        label = i.split(os.path.sep)[-2]
        train_labels.append(label)
        image = load_img(i, target_size=(224, 224), color_mode="grayscale")
        image = apply_match_template(image)
        image = cv.cvtColor(image,cv.COLOR_GRAY2RGB)
        train_data.append(image)

    for i in val_images:  # adicionar nosso preprocessamento
        label = i.split(os.path.sep)[-2]
        val_labels.append(label)
        image = load_img(i, target_size=(224, 224), color_mode="grayscale")
        image = apply_match_template(image)
        image = cv.cvtColor(image,cv.COLOR_GRAY2RGB)
        val_data.append(image)

    print("OK!")

    train_data = np.array(train_data, dtype='float32')
    train_labels = np.array(train_labels)

    train_labels = to_categorical(train_labels)
    val_labels = to_categorical(val_labels)

    aug = ImageDataGenerator()

    print("Training model...", end="")

    lr=0.005
    Epochs=20
    BS=64

    opt=SGD(learning_rate=lr, decay=lr)
    base_model = tf.keras.applications.MobileNetV3Large(input_shape=(224, 224, 3),
                                                include_top=False,
                                                weights='imagenet',
                                                    classifier_activation="leaky_relu")
    base_model.trainable = False

    model = Sequential([base_model,
                        Flatten(),
                        Dense(1000, activation='relu'),
                        #Dropout(0.3),
                        Dense(700, activation='relu'),
                        #Dropout(0.3),
                        Dense(500, activation='relu'),
                        #Dropout(0.3),
                        Dense(5, activation='softmax')])#1024,64,0.2
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    # treina, calculando o tempo gasto para isso
    now = time()
    history = model.fit(
        aug.flow(train_data, train_labels, batch_size=BS),
        steps_per_epoch=len(train_data) // BS,
        validation_data=(val_data, val_labels),
        validation_steps=len(val_data) // BS,
        epochs=Epochs,
        class_weight={0:0.50551181, 1:1.10478011, 2: 0.76226913, 3:1.52655218, 4: 6.67976879}
    )
    print(time() - now)

    acc_train = history.history['accuracy']
    acc_val = history.history['val_accuracy']

    epochs = range(1, 101)
    plt.plot(epochs, acc_train, 'g', label='Training accuracy')
    plt.plot(epochs, acc_val, 'b', label='validation accuracy')
    plt.title('Training and Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    # plt.show()
    plt.savefig("results\\accuracy.png")
    model.save('.\\trained_model_mobileNet.h5')
    loadModel()

def predict(method, img):
    """
    :param method: método a ser utilizado para predizer ["XG","DL","SVM"]
    :return:
    """
    if method == "XG":
        #img = img.reshape((1, 224, 224, 3))
        img_table = extractDescriptors(img)
        prediction = XG.predict([img_table])
        return prediction[0]
    elif method == "DL":
        prediction = DL(img.reshape((1, 224, 224, 3)))
    else:
        img_table = extractDescriptors(img)
        prediction = SVM.predict([img_table])
        return prediction[0]

    return prediction.numpy().argmax()

def extractDescriptors(data):
    media, menor = find_distance_between_bones(data)
    count_black = count_black_pixels(data)
    return (media,menor,count_black)

def showResults():
    print("em progresso")
    """
    mostra todos os resultados obtidos pelo métodos
    :param method: XGBoost, DeepLearning,
    :return:
    """
