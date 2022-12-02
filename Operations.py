import os

import cv2 as cv
from keras_preprocessing.image import img_to_array, array_to_img
import tensorflow as tf

SVM = tf.keras.models.load_model('models/SVM.h5')
XG = tf.keras.models.load_model('models/XGBoost.h5')
DL = tf.keras.models.load_model('models/mobileNet.h5')


def matchTemplatePrivate(img):
    METHOD = cv.TM_CCOEFF

    # lê novamente a imagem para evitar dados quebrados
    edged_img = cv.adaptiveThreshold(img, 255,
                                     cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 21, 10)

    img2 = img.copy()

    # carrega template para joelho esquerdo e direito
    template_l = cv.imread("../templates/template_L.png", 0)
    template_r = cv.imread("../templates/template_R.png", 0)

    # encontra contornos
    edged_template_l = cv.adaptiveThreshold(template_r, 255,
                                            cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 21, 10)

    edged_template_r = cv.adaptiveThreshold(template_l, 255,
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
    print(type(image))

    x, y = matchTemplatePrivate(image)
    print(x, y)
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

            # salva esse arquivo na pasta de pre-processados
            if not os.path.isdir(preprocess_path_sub):
                os.mkdir(preprocess_path_sub)

            filename, file_type = file.split(".")
            filename_eq_path = preprocess_path_sub + "\\" + filename + "_equ." + file_type
            filename_eq_flip_path = preprocess_path_sub + "\\" + filename + "_flipped." + file_type

            cv.imwrite(preprocess_path_sub + "\\" + file, img)
            cv.imwrite(filename_eq_path, equ)
            cv.imwrite(filename_eq_flip_path, flipped)



def count_black_pixels(image):
    # print(cropped)
    ret, bw = cv.threshold(image, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    # count non zero
    IMAGE_SIZE = 224 * 224
    response = cv.countNonZero(bw)
    return IMAGE_SIZE - response


def trainXGBoost():
    print("em progresso")


def trainSVM():
    print("em progresso")


def trainDL():
    print("em progresso")


def predict(method, img):
    """
    :param method: método a ser utilizado para predizer ["XG","DL","SVM"]
    :return:
    """
    prediction = ""

    if method == "XG":
        prediction = XG.predict(img)
    elif method == "DL":
        prediction = DL.predict(img)
    else:
        prediction = SVM.predict(img)

    return prediction


def showResults():
    print("em progresso")
    """
    mostra todos os resultados obtidos pelo métodos
    :param method: XGBoost, DeepLearning,
    :return:
    """
