import numpy as np
from sklearn.model_selection import train_test_split
from dataset_gen import generate_dataset
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tore_gen import tore_view, tore
import os

import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras import backend as K
from tensorflow.keras.constraints import MaxNorm, MinMaxNorm, NonNeg, UnitNorm
import keras
from model import get_rayons, get_all_parameters, get_offsets, get_angles

# ============ Paramétrage
nb_tot = 50000 # Nombre de dataset à générer
nb_point = -1 # -1 pour ne rien générer, mettre nb_tot sinon
parameter_act = 1 # -1 pour non et 1 pour oui (pour générer les datasets suivant les paramètres demandés)
select = 0.9 # 0.X Sur un dataset de nb_tot datasets mettre à minima 0.9!!!

# ------------------ Paramétrage de la génération des datasets ------------
# Les bornes positives et négatives du décalage
x0low = -3
x0high = 3
y0low = -3
y0high = 3
z0low = -3
z0high = 3

# Les bornes positives et négatives des rayons
r1low = 1.5 #r1 grand rayons
r1high = 3
r2low = 0.01
r2high = 1.5

# Les bornes positives et négatives de l'orientation : Euler Z-X-Z (al, be, ga)
allow = -2*np.pi
alhigh = 2*np.pi
below = -2*np.pi
behigh = 2*np.pi
galow = -2*2*np.pi
gahigh = 2*np.pi

# Mode du bruit, "none", "expo", "normal", "uniform" et les paramètres (moy, ecart type)
mode = "normal"
first_p = 0
second_p = 0.2

# Bruit des capteurs coef 
ampl = 0.1
# Sauvegarder les figures des tores générer (False pour non, recommandé)
show = False
save_fig = False
# Nombre de point (!! élevé au carré par la suite) du dataset
Ech = 7

Parameters = [x0low, x0high, y0low, y0high, z0low, z0high, r1low, r1high, r2low, r2high, allow, alhigh, below, behigh, galow, gahigh, mode, first_p, second_p, ampl, save_fig]
Parameters_dict = {"x0low":x0low, "x0high":x0high, "y0low":y0low, "y0high":y0high, "z0low":z0low, "z0high":z0high, "r1low":r1low, "r1high":r1high, "r2low":r2low, "r2high":r2high, "allow":allow, "alhigh":alhigh, "below":below, "behigh":behigh, "gallow":galow, "gahigh":gahigh, "mode":mode, "Moy":first_p, "Std":second_p, "Noise_coef":ampl, "Save_fig":save_fig}

# -------------------------------------------------------------------------
# =========================

print("Génération de dataset: ")
if(nb_point == -1):
    print("Non")
else:
    print("Oui, soit {:d} datasets.".format(nb_point))

if(parameter_act == -1):
    print("Paramètres par défaut.")
    print('=================================================')
    print("r1 = np.random.uniform(low=5, high=20)")
    print("r2 = np.random.uniform(low=0.1, high=5)")
    print("alpha = np.random.uniform(low=0, high=2*np.pi)")
    print("beta = np.random.uniform(low=0, high=2*np.pi)")
    print("gamma = np.random.uniform(low=0, high=2*np.pi)")
    print("x0 = np.random.uniform(low=-10, high=10)")
    print("y0 = np.random.uniform(low=-10, high=10)")
    print("z0 = np.random.uniform(low=-10, high=10)")
    print('=================================================')
    for i in range(nb_tot):
        generate_dataset(nb=nb_point, i=i, show=show)
else:
    print('=================================================')
    print('Paramètres')
    for key, value in Parameters_dict.items():
        print("{:<10} : {:<10}".format(key, value))
    print('=================================================')
    for i in range(nb_tot):
        generate_dataset(nb=nb_point, i=i, Ech=Ech, parameters=Parameters, show=show)

# ============================ Chore ================================

model_all = get_all_parameters(nb_tot = nb_tot, Ech=Ech, epoch=250, batch=516)
model_r = get_rayons(nb_tot = nb_tot, Ech=Ech, epoch=250, batch=516)
model_off = get_offsets(nb_tot = nb_tot, Ech=Ech, epoch=250, batch=516)
model_agl = get_angles(nb_tot = nb_tot, Ech=Ech, epoch=250, batch=516)

# ====================================================================

n = int(nb_tot*select)
#Vérification du modèle :
generate_dataset(nb=1, i=0, Ech=Ech, parameters=Parameters, show=show, path="TorusNtest", info=True)

data = np.load("TorusNtest0.npy")
# Séparation du dataset en X (entrée) et Y (sortie)
X = data[0,:3*Ech**2] # (x ,y ,z)
X = X.reshape((1,3,Ech**2))

# ================== Pour les diverses modèle =====================
y_all = data[:,3*Ech**2:].flatten() # pour les autres paramètres

# pour les autres paramètres
y_verif = model_all.predict(X)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

y_verif = y_verif.flatten()
X_t = tore_view(r1=y_verif[0], r2=y_verif[1], alpha=y_verif[2], beta=y_verif[3], gamma=y_verif[4], x0=y_verif[5], y0=y_verif[6], z0=y_verif[7], nbp=30)

ax.scatter(X[0,0, :], X[0,1, :], X[0,2, :], marker='o', color="blue", alpha=0.2, label="Initial")
ax.scatter(X_t[0, :], X_t[1, :], X_t[2, :], marker='o', color="red", alpha=0.2, label="IA + filtre")
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_xlim([-6, 6])
ax.set_ylim([-6, 6])
ax.set_zlim([-6, 6])
ax.set_title('Torus general')
ax.legend()

k = nb_tot-n
nom = ["r1", "r2", "alpha", "beta", "gamma", "x0", "y0", "z0", "mse", "mae"]

# Graphique des erreurs
Y_view = np.zeros((10,k))
g=0
for i in range(n,nb_tot):
    data = np.load("dataset/torus_datasetN"+str(i)+".npy")
    X = data[:,:3*Ech**2] # (x ,y ,z)
    X = X.reshape((1,3,Ech**2))
    
    y_t = data[:,3*Ech**2:].flatten().reshape(1,-1) # pour les autres paramètres
    y_exp = model_all.predict(X)
    y_exp = y_verif.flatten().reshape(1,-1)

    Y_view[:8,g] = y_t-y_exp

    Y_view[8:,g] = [mean_squared_error(y_t, y_exp), mean_squared_error(y_t, y_exp)]
    g+=1
print("Génération all parameters")
for i in range(8):
    print("{:<10} {:<10.3f} {:<10.3f} {:<10.3f}".format(nom[i], y_all[i], y_verif[i], abs(y_all[i]-y_verif[i])))

# ===================================================

y_r = data[:,3*Ech**2:3*Ech**2+2].flatten() # pour les autres paramètres

# pour les autres paramètres
y_verif = model_r.predict(X)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

y_verif = y_verif.flatten()
X_t = tore_view(r1=y_verif[0], r2=y_verif[1], nbp=30)
k = nb_tot-n

# Graphique des erreurs
Y_view = np.zeros((4,k))
g=0
for i in range(n,nb_tot):
    data = np.load("dataset/torus_datasetN"+str(i)+".npy")
    X = data[:,:3*Ech**2] # (x ,y ,z)
    X = X.reshape((1,3,Ech**2))
    
    y_t = data[:,3*Ech**2:3*Ech**2+2].flatten().reshape(1,-1) # pour les autres paramètres
    y_exp = model_all.predict(X)
    y_exp = y_verif.flatten().reshape(1,-1)

    Y_view[:2,g] = y_t-y_exp

    Y_view[2:,g] = [mean_squared_error(y_t, y_exp), mean_squared_error(y_t, y_exp)]
    g+=1

print("Génération test rayon")
header = ["paramètre", "Théorique", "Expérimentale", "Delta"]
print("{:<10} {:<10} {:<10} {:<10}".format(header[0], header[1], header[2], header[3]))

for i in range(2):
    print("{:<10} {:<10.3f} {:<10.3f} {:<10.3f}".format(nom[i], y_r[i], y_verif[i], abs(y_r[i]-y_verif[i])))

# ===================================================

y_a = data[:,3*Ech**2+2:3*Ech**2+2+3].flatten() # pour les autres paramètres

# pour les autres paramètres
y_verif = model_agl.predict(X)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

y_verif = y_verif.flatten()
X_t = tore_view(alpha=y_verif[0], beta=y_verif[1], gamma=y_verif[2], nbp=30)
k = nb_tot-n

# Graphique des erreurs
Y_view = np.zeros((5,k))
g=0
for i in range(n,nb_tot):
    data = np.load("dataset/torus_datasetN"+str(i)+".npy")
    X = data[:,:3*Ech**2] # (x ,y ,z)
    X = X.reshape((1,3,Ech**2))
    
    y_t = data[:,3*Ech**2+2:3*Ech**2+2+3].flatten().reshape(1,-1) # pour les autres paramètres
    y_exp = model_all.predict(X)
    y_exp = y_verif.flatten().reshape(1,-1)

    Y_view[:3,g] = (y_t-y_exp)

    Y_view[3:,g] = [mean_squared_error(y_t, y_exp), mean_squared_error(y_t, y_exp)]
    g+=1

print("Génération test angles")
header = ["paramètre", "Théorique", "Expérimentale", "Delta"]
print("{:<10} {:<10} {:<10} {:<10}".format(header[0], header[1], header[2], header[3]))

for i in range(3):
    print("{:<10} {:<10.3f} {:<10.3f} {:<10.3f}".format(nom[i+2], y_a[i]*180/np.pi, y_verif[i]*180/np.pi, abs(y_a[i]-y_verif[i])*180/np.pi))

# ===================================================

y_off = data[:,3*Ech**2+3:3*Ech**2+2+3+3].flatten() # pour les autres paramètres

# pour les autres paramètres
y_verif = model_agl.predict(X)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

y_verif = y_verif.flatten()
k = nb_tot-n

# Graphique des erreurs
Y_view = np.zeros((5,k))
g=0
for i in range(n,nb_tot):
    data = np.load("dataset/torus_datasetN"+str(i)+".npy")
    X = data[:,:3*Ech**2] # (x ,y ,z)
    X = X.reshape((1,3,Ech**2))
    
    y_t = data[:,3*Ech**2+2:3*Ech**2+2+3].flatten().reshape(1,-1) # pour les autres paramètres
    y_exp = model_all.predict(X)
    y_exp = y_verif.flatten().reshape(1,-1)

    Y_view[:3,g] = y_t-y_exp

    Y_view[3:,g] = [mean_squared_error(y_t, y_exp), mean_squared_error(y_t, y_exp)]
    g+=1

print("Génération test offset")
header = ["paramètre", "Théorique", "Expérimentale", "Delta"]
print("{:<10} {:<10} {:<10} {:<10}".format(header[0], header[1], header[2], header[3]))

for i in range(3):
    print("{:<10} {:<10.3f} {:<10.3f} {:<10.3f}".format(nom[i+2+3], y_a[i], y_verif[i], abs(y_a[i]-y_verif[i])))

# ===================================================

data = np.load("TorusNtest0.npy")
# Séparation du dataset en X (entrée) et Y (sortie)
X = data[0,:3*Ech**2] # (x ,y ,z)
X = X.reshape((1,3,Ech**2))

# pour les autres paramètres
y_r = model_r.predict(X)
y_a = model_agl.predict(X)
y_off = model_off.predict(X)

y_verif = [item for sublist in [y_r, y_a, y_off] for item in np.ravel(sublist)]
print(y_verif)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

X_t = tore_view(r1=y_verif[0], r2=y_verif[1], alpha=y_verif[2], beta=y_verif[3], gamma=y_verif[4], x0=y_verif[5], y0=y_verif[6], z0=y_verif[7], nbp=30)

ax.scatter(X[0,0, :], X[0,1, :], X[0,2, :], marker='o', color="blue", alpha=0.2, label="Initial")
ax.scatter(X_t[0, :], X_t[1, :], X_t[2, :], marker='o', color="red", alpha=0.2, label="IA + filtre")
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_xlim([-6, 6])
ax.set_ylim([-6, 6])
ax.set_zlim([-6, 6])
ax.set_title('Torus all parameters')
ax.legend()


plt.show()

'''
#Sauver le modèle
nom = input("Voulez-vous sauver le modèle (o/n): ")
if(nom == "o"):
    model_all.save("modele/"+input("Nom du modèle (complet): ")+'_'+str(Ech**2)+'.h5') '''