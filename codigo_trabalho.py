import numpy as np
import numpy.linalg as linalg
from PIL import Image
import glob
import math
import os
import cv2

# Funçao para aplicar o metodo PCA
def pca(X, confianca):
    # Media do dataset
    mean = np.mean(X,0)
    # Centrar os dados
    phi = X - mean
    # Calcular os vetores e valores proprios atraves do SVD
    vetores_proprios, sigma, variancia = np.linalg.svd(phi.transpose(), full_matrices=False)
    valores_proprios= sigma*sigma   
    # Ordenacao dos valores próprios
    indice = np.argsort(-valores_proprios)
    valores_proprios = valores_proprios[indice]
    vetores_proprios = vetores_proprios[:,indice] 
    # Determinar o numero de vectores proprios 
    k = 0
    traco = np.sum(valores_proprios)
    while(np.sum(valores_proprios[:k])/traco < confianca):
        k = k+1  
    vetores_proprios = vetores_proprios[:,0:k]
    return k, valores_proprios,vetores_proprios, phi, mean, variancia


def coeficientes_projecao(phi,vetores_proprios,M):
    coeficientes = [np.dot(phi[i],vetores_proprios) for i in range(M)]
    return coeficientes





def testar(input_img , mean,vetores_proprios ,valores_proprios , M , coeficientes_proj , distance ): 
    dist = [] 
    # Centrar a imagem na média
    gamma = np.array(input_img.getdata())
    test_phi = gamma - mean
    
    # Calcular os coeficientes da projeçao do input
    test_coef_proj = np.dot(test_phi , vetores_proprios)
    
    if distance == "euclidian":
        dist = [euclidian(coeficientes_proj[i], test_coef_proj) for i in range (M)]
        d_min = round(np.min(dist),2)
        d_max = round(np.max(dist),2)
        limit = 7000
    elif distance == "mahalanobis" :
        dist = mahalanobis(coeficientes_proj , test_coef_proj , valores_proprios , vetores_proprios.shape [1])
        d_min = round(np.min(dist),4)
        d_max = round(np.max(dist),4)
        limit = 0.45
    else: 
        print("Distancia invalida.")
        return (-1)

    if d_min < limit:
        print('Imagem nr.: '+str(np.argmin(dist))+'\n'+'Distancia minima: '+ str(d_min)+ '\nDistancia máxima: '+ str(d_max)+'\n')
        return dist, test_coef_proj
    else: 
        print('Falhou no reconhecimento.')
        return [],[]


# Calculo da distância euclidiana
def euclidian(x, y):
    if x.size != y.size:
        return (-1)
    z = y - x
    distancia = math.sqrt(sum(z**2))
    return distancia

#Calculo da distância de Mahalanobis
def mahalanobis(x, y, valores_proprios, k):
    if len(x[0]) != len(y):
        return (-1) 
    N = len(x)
    distancia=[]
    for i in range(N):
        distancia.append(np.sum(np.divide((x[i]-y)**2, valores_proprios[:k]))) 
    return distancia