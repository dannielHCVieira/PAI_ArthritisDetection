import cv2

# 
def shape_selection(event, x, y, flags, param):
	global ref_point, crop

	# se o botão esquerdo do mouse foi clicado, registra o início
	# coordenadas (x, y) indicam que o corte está sendo executado
	if event == cv2.EVENT_LBUTTONDOWN:
		ref_point = [(x, y)]

	# verifica se o botão esquerdo do mouse foi liberado
	elif event == cv2.EVENT_LBUTTONUP:
		# registra as coordenadas finais (x, y) e indica que o corte foi finalizado
		ref_point.append((x, y))

		# desenha um retangulo
		cv2.rectangle(image, ref_point[0], ref_point[1], (0, 255, 0), 2)
		cv2.imshow("image", image)

def create_environment(imageLoaded):
	# inicializar a lista de pontos de referência
	global ref_point
	ref_point = []
	crop = False

	global image
	# le a imagem, faz o clone e configura a função de retorno de chamada do mouse
	image = cv2.imread(imageLoaded)
	clone = image.copy()
	cv2.namedWindow("image")
	cv2.setMouseCallback("image", shape_selection)





