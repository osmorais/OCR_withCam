import picamera
import time
from PIL import Image
import pytesseract as pyt
import os
import cv2
from imutils.object_detection import non_max_suppression
import numpy as np
import argparse


# Capturar imagem que será processada
camera = picamera.PiCamera()
print("[INFO] Sorria para a foto!...")
camera.start_preview()
for i in range(10):
    #camera.saturation = i * 10
    #camera.contrast = i * 10
    #if i%2 == 0 :
    #    camera.hflip = True
    #else :
    #    camera.hflip = False
    time.sleep(0.5)
camera.stop_preview()
#camera.capture("foto.jpg")
print("[INFO] Imagem capturada...")


# Carregar a imagem e pegar suas dimensoes
imagem = cv2.imread("foto.jpg")
original = imagem.copy()
(H, W) = imagem.shape[:2]


# definir a nova largura e altura e determinar a proporção da mudança
# para largura e altura
(novaLargura,novaAltura) = (320,320)
rLargura = W / float(novaLargura)
rAltura = H / float(novaAltura)

#redimensionar a imagem para os novos valores
imagem = cv2.resize(imagem,(novaAltura,novaLargura))
(H,W) = imagem.shape[:2]

# definis os dois layernames para o modelo de detector EAST que
# esta interessado ​primeiramente nas probabilidades de haver um texto na imagem e
# em seguida para derivar as coordenadas da caixa delimitadora do texto.

#o primeiro layername devolve a propabilidade de uma regiao conter algum texto
#o segundo layername devolve um mapa de cordenadas onde esses textos podem estar
layerNames = ["feature_fusion/Conv_7/Sigmoid","feature_fusion/concat_3"]

# carregar o detector EAST
print("[INFO] Carregando o detector de texto EAST...")
net = cv2.dnn.readNet("frozen_east_text_detection.pb")

# construir um blob a partir da imagem e fazer uma
# passagem direta do modelo para obter os dois conjuntos de camadas de saída
blob = cv2.dnn.blobFromImage(imagem, 1.0, (W, H),
	(123.68, 116.78, 103.94), swapRB=True, crop=False)
inicio = time.time()
net.setInput(blob)
print("[INFO] Localizando cordenadas de possiveis textos na imagem...")
(scores, areas) = net.forward(layerNames)
fim = time.time()

# show timing information on text prediction
print("[INFO] A detecção de texto levou {:.6f} segundos".format(fim - inicio))

# pegar o numero de linhas e colunas do score coletado pelo layer, depois
# demarcar as areas reconhecidas com retangulos
(numeroLinhas, numeroColunas) = scores.shape[2:4]

#coletar as cordenadas de cada possivel texo
cordenadas = []
#coletar a possibilidade do espaco delimitado ser um texto
confiabilidade = []

# loop por cada linha
for y in range(0, numeroLinhas):

    #extrair as pontuações(probabilidades), seguidas pelos
    # dados geométricos usados ​​para derivar possíveis coordenadas
    # da caixa delimitadora que circundam o texto
    scoresData = scores[0, 0, y]
    xData0 = areas[0, 0, y]
    xData1 = areas[0, 1, y]
    xData2 = areas[0, 2, y]
    xData3 = areas[0, 3, y]
    anguloData = areas[0, 4, y]

    # loop por cada coluna
    for x in range(0, numeroColunas):
        # se o score não tiver pontuacao suficiente, ignorar o processo
        if scoresData[x] < 0.5:
            continue

        #calcular o fator de deslocamento, pois nossos mapas
        # de recursos resultantes serão 4x menores que a imagem de entrada
        (deslocamentoX, deslocamentoY) = (x * 4.0, y * 4.0)

        # extrair o ângulo de rotação para a previsão e depois
        # calcular o seno e o cosseno
        angulo = anguloData[x]
        cos = np.cos(angulo)
        sen = np.sin(angulo)

        # usar a geometria para delimitar a largura e altura das caixas
        # delimitadoras dos textos
        h = xData0[x] + xData2[x]
        w = xData1[x] + xData3[x]

        #calcular ambos começos e fins, cordenadas (x,y) para
        # deteccao das caixas delimitadoras de texto
        finalX = int(deslocamentoX + (cos * xData1[x]) + (sen * xData2[x]))
        finalY = int(deslocamentoY - (sen * xData1[x]) + (cos * xData2[x]))
        inicioX = int(finalX - w)
        inicioY = int(finalY - h)

        # adicionar os dados coletados para a lista de caixas contidas com textos detectados
        cordenadas.append((inicioX, inicioY, finalX, finalY))
        confiabilidade.append(scoresData[x])

# aplicar supressão não máxima para suprimir caixas delimitadoras fracas e sobrepostas
caixas = non_max_suppression(np.array(cordenadas), probs=confiabilidade)

# loop na lista de caixxas delimitadores de texto encontradas
margemErro = 40
results = []
for (inicioX, inicioY, finalX, finalY) in caixas:

    # escala das cordenadas das caixas delimitadores baseado
    # em sua respectiva proporcao
    inicioX = int(inicioX * rLargura)
    inicioY = int(inicioY * rAltura)
    finalX = int(finalX * rLargura)
    finalY = int(finalY * rAltura)

    finalX += margemErro + margemErro
    finalY += margemErro + margemErro
    inicioY -= margemErro
    inicioX -= margemErro
    # desenhar as caixas na imagem
    cv2.rectangle(original, (inicioX, inicioY),(finalX, finalY),(0, 255, 0), 2)


    # in order to obtain a better OCR of the text we can potentially
    # apply a bit of padding surrounding the bounding box -- here we
    # are computing the deltas in both the x and y directions
    #dX = int((finalX - inicioX) * padding)
    #dY = int((finalY - inicioY) * padding)

    # apply padding to each side of the bounding box, respectively
    #inicioX = max(0, inicioX - dX)
    #inicioY = max(0, inicioY - dY)
    #finalX = min(W, finalX + (dX * 2))
    #finalY = min(H, finalY + (dY * 2))

    # extract the actual padded ROI
    roi = original[inicioY:finalY, inicioX:finalX]

    roigray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    limiar, imgLimiar = cv2.threshold(roigray, 50, 255, cv2.THRESH_BINARY)

    imagemDesfocada = cv2.GaussianBlur(imgLimiar, (7, 7), 0)

    laplace = cv2.Laplacian(imagemDesfocada, cv2.CV_16S, ksize=3)

    # in order to apply Tesseract v4 to OCR text we must supply
    # (1) a language, (2) an OEM flag of 4, indicating that the we
    # wish to use the LSTM neural net model for OCR, and finally
    # (3) an OEM value, in this case, 7 which implies that we are
    # treating the ROI as a single line of text
    #config = ("outputbase digits -l eng --oem 1 --psm 7")
    config = ("-l eng --oem 1 --psm 7")
    text = pyt.image_to_string(laplace, config=config)

    # add the bounding box coordinates and OCR'd text to the list
    # of results

    limiar = cv2.convertScaleAbs(limiar)
    laplace = cv2.convertScaleAbs(laplace)
    results.append(((inicioX, inicioY, finalX, finalY), text, imagemDesfocada, limiar,laplace))
# sort the results bounding box coordinates from top to bottom
results = sorted(results, key=lambda r: r[0][1])

print("[INFO] Reconhecimento realizado...")
print("[INFO] Exibindo dados calculados...")
# loop over the results
for ((startX, startY, endX, endY), text, imagemDesfocada, limiar,laplace) in results:
    # display the text OCR'd by Tesseract
    print("OCR TEXT")
    print("========")
    print("{}\n".format(text))

    # strip out non-ASCII text so we can draw the text on the image
    # using OpenCV, then draw the text and a bounding box surrounding
    # the text region of the input image
    text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
    output = original.copy()
    cv2.rectangle(output, (startX, startY), (endX, endY),
                  (0, 0, 255), 2)
    cv2.putText(output, text, (startX, startY - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    output = cv2.resize(output, None, fx=0.65, fy=0.65, interpolation=cv2.INTER_CUBIC)
    imagemDesfocada = cv2.resize(imagemDesfocada, None, fx=0.65, fy=0.65, interpolation=cv2.INTER_CUBIC)
    # show the output image
    cv2.imshow("Text Detection", output)
    cv2.imshow("Text Detection 2", imagemDesfocada)
    cv2.imshow("Text Detection 3", limiar)
    cv2.imshow("Text Detection 4", laplace)
    cv2.waitKey(0)


