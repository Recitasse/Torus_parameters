from tensorflow import keras
import os
import csv
import numpy as np
from tore_gen import tore_view
import matplotlib.pyplot as plt
import random

# Afficher les Données disponnibles
print("---------------------------------------")
for file in os.listdir("Data/"):
    print(file)
print("\n")

X = []; Y = []; Z = []
with open("Data/"+input("Nom du fichier de donnée: ")+'.csv','r') as d:
    lec = csv.reader(d, delimiter=input("donner un délimiteur: "))
    next(lec) #on passe la ligne des headers
    for ligne in lec:
        X.append(eval(ligne[0]))
        Y.append(eval(ligne[1]))
        Z.append(eval(ligne[2]))
n = np.size(X)
X_m = np.zeros((3,np.size(X)))
X_m[0,:] = [float(x) for x in X[:]]
X_m[1,:] = [float(x) for x in Y[:]]
X_m[2,:] = [float(x) for x in Z[:]]

# Chargement du modèle
print("---------------------------------------")
for file in os.listdir("modele/"):
    if file.endswith(".h5"):
        print(file)
print("\n")

nom=input("Donner le nom du modele: ")
number = int(''.join(filter(str.isdigit, nom)))
if(np.size(X_m[0,:]) < number):
    print("Impossible de faire tourner le modèle, demande un minimum de "+str(number)+" points.")
    exit()
elif(np.size(X_m[0,:]) > number):
    indices = np.arange(np.size(X_m[0,:]))
    np.random.shuffle(indices)
    X_m = X_m[:, indices]  
    X_m = X_m[:, :number]  
    X_m = np.reshape(X_m, (1,3, number))

model = keras.models.load_model("modele/"+nom+'.h5')
y_verif = model.predict(X_m)
y_verif = y_verif.flatten()

print("============================================================")
header = ["paramètre", "Expérimentale"]
print("{:<10} {:<10}".format(header[0], header[1]))
nom = ["r1", "r2", "alpha", "beta", "gamma", "x0", "y0", "z0"]
for i in range(8):
    print("{:<10} {:<10.3f}".format(nom[i], y_verif[i]))
print("============================================================")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

X_t = tore_view(r1=y_verif[0], r2=y_verif[1], alpha=y_verif[2], beta=y_verif[3], gamma=y_verif[4], x0=y_verif[5], y0=y_verif[6], z0=y_verif[7], nbp=75)

ax.scatter(X_m[0, 0, :], X_m[0, 1, :], X_m[0, 2, :], marker='o', color="blue", alpha=0.7, label="Initial")
ax.scatter(X_t[0, :], X_t[1, :], X_t[2, :], marker='o', color="red", alpha=0.2, label="IA + filtre")
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_xlim([-6, 6])
ax.set_ylim([-6, 6])
ax.set_zlim([-6, 6])
ax.set_title('Torus')
ax.legend()
plt.show()

