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


def count_black_pixels(image):
    # print(cropped)
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

    test_labels = np.array(test_labels, dtype=object)
    test_labels_int = test_labels.astype(np.dtype(np.int_))

    xgb_model = pickle.load(open('models/XGBoost.h5','rb'))

    xgb_predictions = xgb_model.predict(test_data_table)

    fig = plot_confusion_matrix(xgb_model, test_data_table, test_labels_int, cmap='Blues')
    plt.xlabel('', fontsize=18)
    plt.ylabel('', fontsize=18)
    plt.savefig("results\\svm_cm.png")

    report = classification_report(test_labels_int, xgb_predictions, target_names=["0", "1", "2", "3", "4"])
    return report

def trainXGBoost(path_train):
    train_dataset = path_train

    train_images = list(paths.list_images(train_dataset))

    train_data = []
    train_labels = []

    for i in train_images:  # adicionar nosso preprocessamento
        label = i.split(os.path.sep)[-2]
        train_labels.append(label)
        image = load_img(i, target_size=(224, 224), color_mode="grayscale")
        image = processImage(image)
        train_data.append(image)

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
    
    now = time()
    xgb_model = XGBClassifier(n_estimators = 650,
                      max_depth = 10,
                      learning_rate = 0.01,
                      subsample = 1,
                      random_state = 0
                     )
    now = time()
    xgb_history = xgb_model.fit(train_data_np, train_labels_int, verbose=False)
    print(time() - now)
    xgb_filename = '../models/XGBoost.h5'
    pickle.dump(xgb_model, open(xgb_filename,
                            'wb'))

def testSVM(path_test):
    test_dataset = path_test

    test_images=list(paths.list_images(test_dataset))

    test_data=[]
    test_labels=[]

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
    
    svm_model = pickle.load(open('models/SVM.h5','rb'))

    svm_predict = svm_model.predict(test_data_table)

    fig = plot_confusion_matrix(svm_model, test_data_table, test_labels, cmap='Blues')
    plt.xlabel('', fontsize=18)
    plt.ylabel('', fontsize=18)
    plt.savefig("results\\svm_cm.png")

    report = classification_report(test_labels ,svm_predict, target_names=["0", "1", "2", "3", "4"])
    return report

def trainSVM(path_train):
    train_dataset = path_train

    train_images = list(paths.list_images(train_dataset))

    train_data = []
    train_labels = []

    for i in train_images:  # adicionar nosso preprocessamento
        label = i.split(os.path.sep)[-2]
        train_labels.append(label)
        image = load_img(i, target_size=(224, 224), color_mode="grayscale")
        image = processImage(image)
        train_data.append(image)

    train_data = np.array(train_data, dtype='uint8')
    train_labels = np.array(train_labels)

    # ainda não acabou
    train_data_table = []
    for data in train_data:
        media, menor = find_distance_between_bones(data)
        count_black = count_black_pixels(data)
        train_data_table.append((media,menor,count_black))

    now = time()
    svm_model = SVC(kernel='rbf', class_weight='balanced').fit(train_data_table, train_labels)
    print(now - time())

    svm_filename = '../models/SVM.h5'
    pickle.dump(svm_model, open(svm_filename, 'wb'))

def testDL(path_test):
    test_dataset = path_test

    test_images = list(paths.list_images(test_dataset))

    test_data = []
    test_labels = []


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

    BS = len(test_data)//10

    aug = ImageDataGenerator()
    print(len(test_data))

    predict = DL.predict(aug.flow(test_data), batch_size=BS)
    predict = np.argmax(predict, axis=1)
    print(predict)
    print(test_labels)
    report = classification_report(test_labels.argmax(axis=1), predict, target_names=["0", "1", "2", "3", "4"])

    cm = confusion_matrix(y_true=test_labels.argmax(axis=1), y_pred=predict)
    fig, ax = plot_confusion_matrix(conf_mat=cm,
                                    figsize=(6, 6),
                                    class_names=["0", "1", "2", "3", "4"],
                                    # cmap='Greys',

                                    norm_colormap=matplotlib.colors.LogNorm())

    plt.xlabel('', fontsize=18)
    plt.ylabel('', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    # plt.show()
    plt.savefig("results\\cm.png")
    return report


def trainDL(path_train, path_val):
    train_dataset = path_train
    val_dataset = path_val

    train_images = list(paths.list_images(train_dataset))
    val_images = list(paths.list_images(val_dataset))

    train_data = []
    train_labels = []

    val_data = []
    val_labels = []

    for i in train_images:  # carrega imagens e preprocessa
        label = i.split(os.path.sep)[-2]
        train_labels.append(label)
        image = load_img(i, target_size=(224, 224), color_mode="grayscale")
        image = apply_match_template(image)
        train_data.append(image)

    for i in val_images:  # adicionar nosso preprocessamento
        label = i.split(os.path.sep)[-2]
        val_labels.append(label)
        image = load_img(i, target_size=(224, 224), color_mode="grayscale")
        image = apply_match_template(image)
        val_data.append(image)

    train_data = np.array(train_data, dtype='float32')
    train_labels = np.array(train_labels)

    train_labels = to_categorical(train_labels)
    val_labels = to_categorical(val_labels)

    aug = ImageDataGenerator()

    lr = 0.000005
    Epochs = 100
    BS = 128

    opt = Adam(learning_rate=lr)
    base_model = tf.keras.applications.MobileNetV3Small(input_shape=(224, 224, 3),
                                                        include_top=False,
                                                        weights='imagenet')
    base_model.trainable = False

    # cria modelo especificado
    model = Sequential([base_model,
                        Flatten(),
                        Dense(1024, activation='leaky_relu'),
                        Dropout(0.5),
                        Dense(512, activation='leaky_relu'),
                        Dropout(0.5),
                        Dense(512, activation='leaky_relu'),
                        Dropout(0.5),
                        Dense(5, activation='softmax')])  # 1024,64,0.2
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt, metrics=['accuracy'])

    # treina, calculando o tempo gasto para isso
    now = time()
    history = model.fit(
        aug.flow(train_data, train_labels, batch_size=BS),
        steps_per_epoch=len(train_data) // BS,
        validation_data=(val_data, val_labels),
        validation_steps=len(val_data) // BS,
        epochs=Epochs,
        class_weight={0: 1, 1: 4, 2: 2, 3: 4, 4: 9}
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


def predict(method, img):
    """
    :param method: método a ser utilizado para predizer ["XG","DL","SVM"]
    :return:
    """
    if method == "XG":
        prediction = XG.predict(img.reshape((1, 224, 224, 3)))
        print(prediction.numpy().argmax())
    elif method == "DL":
        prediction = DL(img.reshape((1, 224, 224, 3)))
        print(prediction.numpy().argmax())
    else:
        prediction = SVM.predict(img.reshape((1, 224, 224, 3)))
        print(prediction.numpy().argmax())

    return prediction.numpy().argmax()


def showResults():
    print("em progresso")
    """
    mostra todos os resultados obtidos pelo métodos
    :param method: XGBoost, DeepLearning,
    :return:
    """
