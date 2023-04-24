import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from dataset_gen import generate_dataset
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tore_gen import tore_view, tore
import os

import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras import backend as K
import keras

# ============ Paramétrage
nb_tot = 10000 # Nombre de dataset à générer
nb_point = -1 # -1 pour ne rien générer, mettre nb_tot sinon
parameter_act = 1 # -1 pour non et 1 pour oui (pour générer les datasets suivant les paramètres demandés)
select = 1 # 0.X Sur un dataset de nb_tot datasets

# ------------------ Paramétrage de la génération des datasets ------------
# Les bornes positives et négatives du décalage
x0low = -2 
x0high = 2
y0low = -2
y0high = 2
z0low = -2
z0high = 2

# Les bornes positives et négatives des rayons
r1low = 2 #r1 grand rayons
r1high = 5
r2low = 0.025
r2high = 1.9

# Les bornes positives et négatives de l'orientation : Euler Z-X-Z (al, be, ga)
allow = -0.2
alhigh = 0.2
below = -0.2
behigh = 0.2
galow = -0.2
gahigh = 0.2

# Mode du bruit, "none", "expo", "normal", "uniform" et les paramètres (moy, ecart type)
mode = "uniform"
first_p = -0.05
second_p = 0.05

# Bruit des capteurs coef 
ampl = 0.2
# Sauvegarder les figures des tores générer (False pour non, recommandé)
show = False
save_fig = False
# Nombre de point (!! élevé au carré par la suite) du dataset
Ech = 13

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
    for key, value in Parameters_dict.items():
        print("{:<10} : {:<10}".format(key, value))
    print('=================================================')
    for i in range(nb_tot):
        generate_dataset(nb=nb_point, i=i, Ech=Ech, parameters=Parameters, show=show)



file_names = os.listdir("dataset/")
file_names = sorted(file_names)
n = int(len(file_names) * select)
print(n)
full_dataset = np.concatenate([np.load("dataset/"+f) for f in file_names[:n]])

# Séparation du dataset en X (entrée) et Y (sortie)
X = full_dataset[:,:3*Ech**2] # (x ,y ,z)
print(X.shape)
X = X.reshape((n, 3, Ech**2))
y = full_dataset[:,3*Ech**2:] # pour les autres paramètres

print(X.shape)
print(y.shape)

# randomiser les datasets
np.random.seed(42)
melange_indices = np.random.permutation(len(full_dataset)) # randomise les indices
X_melange = X[melange_indices]
y_melange = y[melange_indices]

# Séparation des datasets d'entrainement et de tests
X_train, X_test, y_train, y_test = train_test_split(X_melange, y_melange, test_size=0.2, random_state=42) # 20% pour le dataset test

# Création du modèle ======================

def Angle(p):
    
    # Pour les angles
    p3, p4, p5 = p[:, 2], p[:, 3], p[:, 4]
    p3 = K.clip(p3, -2*np.pi, 2*np.pi)
    p4 = K.clip(p4, -2*np.pi, 2*np.pi)
    p5 = K.clip(p5, -2*np.pi, 2*np.pi)
    p3 = K.reshape(p3, (-1, 1))
    p4 = K.reshape(p4, (-1, 1))
    p5 = K.reshape(p5, (-1, 1))

    # Pour les rayons
    p1, p2 = p[:, 0], p[:, 1]


    return K.concatenate([K.expand_dims(K.maximum(p1, p2), axis=-1), K.expand_dims(K.minimum(p1, p2), axis=-1), p3, p4, p5, p[:, 5:]], axis=1)

model = Sequential()
model.add(Dense(256, activation='relu', input_shape=(3,Ech**2)))
model.add(Flatten())
model.add(Dense(128, activation='linear',
                kernel_initializer='uniform',
                bias_initializer='zeros'))
model.add(Dense(32, activation='linear',
                kernel_initializer='uniform',
                bias_initializer='zeros'))
model.add(Dense(16, activation='linear',
                kernel_initializer='uniform',
                bias_initializer='zeros'))
model.add(Dense(16, activation='linear',
                kernel_initializer='uniform',
                bias_initializer='zeros'))
model.add(Dense(32, activation='linear',
                kernel_initializer='uniform',
                bias_initializer='zeros'))
model.add(Dense(8, activation='linear'))

print(model.summary())
# Compiler le modèle

# ============= Optimiseur ================
adam = Adam(lr=0.00135, beta_1=0.75, beta_2=0.8, epsilon=1e-08)
sgd = SGD(lr=0.009, momentum=0.88, nesterov=True)
model.compile(optimizer=adam, loss='mse', metrics=['accuracy'])

# Entraînement
model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_test, y_test))
# Validate the model's performance on a separate set of data
loss, accuracy = model.evaluate(X_test, y_test)
# =========================================

#Vérification du modèle :
generate_dataset(nb=1, i=0, Ech=Ech, parameters=Parameters, show=show, path="TorusNtest")

data = np.load("TorusNtest0.npy")
# Séparation du dataset en X (entrée) et Y (sortie)
X = data[0,:3*Ech**2] # (x ,y ,z)
X = X.reshape((1,3,Ech**2))
y = data[:,3*Ech**2:].flatten() # pour les autres paramètres

# pour les autres paramètres
y_verif = model.predict(X)

print("============================================================")
header = ["paramètre", "Théorique", "Expérimentale", "Delta"]
print("{:<10} {:<10} {:<10}".format(header[0], header[1], header[2], header[3]))
nom = ["r1", "r2", "alpha", "beta", "gamma", "x0", "y0", "z0"]
for i in range(8):
    print("{:<10} {:<10.3f} {:<10.3f} {:<10.3f}%".format(nom[i], y_verif[0, i], y[i], 100*abs((y[i]-y_verif[0, i])/y[i])))
print("============================================================")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

y_verif = y_verif.flatten()
X_t = tore_view(r1=y_verif[0], r2=y_verif[1], alpha=y_verif[2], beta=y_verif[3], gamma=y_verif[4], x0=y_verif[5], y0=y_verif[6], z0=y_verif[7], nbp=Ech*4)

ax.scatter(X[0,0, :], X[0,1, :], X[0,2, :], marker='o', color="blue", alpha=0.7, label="Initial")
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

#Sauver le modèle
nom = input("Voulez-vous sauver le modèle (o/n): ")
if(nom == "o"):
    model.save("modele/"+input("Nom du modèle: ")+'_'+str(Ech**2)+'.h5') 