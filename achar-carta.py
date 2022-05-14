#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt
import cv2
# Numero mínimo de matches para considerar que encontrou o objeto
minKPMatch=30

# Inicializa o SIFT
sift=cv2.SIFT_create()

# Carrega a imagem de referencia na escala de cinza.
refImg=cv2.imread("carta.png",0)

# Calcula os keypoints e Descritores da imagem de referencia
refKP,refDesc = sift.detectAndCompute(refImg,None)

# configura a captura de imagem da webcam
vc=cv2.VideoCapture("q1.mp4")

while True:
    ret, frame = vc.read()

    if not ret:
        break

    # Converte o frame da web para escala de cinza
    frameImg=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    # Calcula os keypoints e Descritores do frame recebido pela webcam
    frameKP, frameDesc = sift.detectAndCompute(frameImg,None)

    # Cria e usa o metodo Força Bruta Matcher
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(refDesc,frameDesc, k=2) 

    # Filtra os matches encontrados em matches (m,n), para obter um resultado mais limpo
    goodMatch=[]
    for m,n in matches:
        if(m.distance < 0.85*n.distance):
            goodMatch.append(m)
       
    # Testa se foram encontrados matches acima do minimo definido
    if(len(goodMatch)> minKPMatch):
        tp=[]
        qp=[]
        for m in goodMatch:
            qp.append(refKP[m.queryIdx].pt) # fornece os indices de um ID e .pt as coordenadas
            tp.append(frameKP[m.trainIdx].pt)
        tp,qp=np.float32((tp,qp))
        
        # o findHomography mapeia os pontos de um plano em outro.
        H,status=cv2.findHomography(qp,tp,cv2.RANSAC,3.0)
        
        # extrai o shape da imagem de referencia
        h,w=refImg.shape
        refBorda=np.float32([[[0,0],[0,h-1],[w-1,h-1],[w-1,0]]])
        # Usa refBorda e a matrix de homografia H para calcular a matrix transformação de pespectiva
        frameBorda=cv2.perspectiveTransform(refBorda,H)
        # polylines desenha poligonos ou qualquer imagem, na cor verde e largura do traço igual a 5.
        cv2.polylines(frame,[np.int32(frameBorda)],True,(0,255,0),2)
        print ("Encontrado bom match - %d/%d"%(len(goodMatch),minKPMatch))
    else:
        print ("Não encontrado bom match - %d/%d"%(len(goodMatch),minKPMatch))

    # Exibe saida da imagem
    cv2.imshow("resultado", frame)
    # Atualiza com um novo frame
    frame = vc.read()

    # ESC para sair do programa
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
vc.release()
cv2.destroyAllWindows()