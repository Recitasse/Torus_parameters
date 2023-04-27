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


# Obtention de tout les paramètres d'un seul coup
def get_all_parameters(nb_tot:int = -1, epoch:int = 250, batch:int = 32, select:float=0.9, parameters: np.ndarray = np.array([-10, 10, -10, 10, -10, 10, 5, 20, 0.1, 5, 0, 2*np.pi, 0, 2*np.pi, 0, 2*np.pi, "uniform", 0.1, -0.1, 0.5, False]), pat:int=3, Ech:int=-1):

    if(Ech == -1):
        print("Veuillez renseigner exactement le nombre d'achantillon.")
        exit()
    if(nb_tot == -1):
        print("Veuillez renseigner exactement le nombre de datasets.")
        exit()
    #========================== Soit le default array ================
    # x0low, x0high, y0low, y0high, z0low, z0high
    # r1low, r1high, r2low, r2high
    # allow, alhigh, below, behigh, galow, gahigh
    # mode, first_p, second_p
    # ampl
    # ================================================================

    x0low = parameters[0]
    x0high = parameters[1]

    y0low = parameters[2]
    y0high = parameters[3]

    z0low = parameters[4]
    z0high = parameters[5]

    r1low = parameters[6]
    r1high = parameters[7]

    r2low = parameters[8]
    r2high = parameters[9]

    allow = parameters[10] 
    alhigh = parameters[11]
    below = parameters[12] 
    behigh = parameters[13]
    galow = parameters[14] 
    gahigh = parameters[15]

    mode = parameters[16] 
    first_p = parameters[17] 
    second_p = parameters[18]
    ampl = parameters[19]
    save_fig = parameters[20]

    Parameters = [x0low, x0high, y0low, y0high, z0low, z0high, r1low, r1high, r2low, r2high, allow, alhigh, below, behigh, galow, gahigh, mode, first_p, second_p, ampl, save_fig]

    # ------------------------------------------------ 

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
    # Dû à la version 1.15 de tensorflow (inapte à la 2.0)
    class SequentialConstraint(tf.keras.constraints.Constraint):
        def __init__(self, constraints):
            self.constraints = constraints

        def __call__(self, w):
            for constraint in self.constraints:
                w = constraint(w)
            return w

    model = Sequential()
    model.add(Dense(256, activation='sigmoid', input_shape=(3,Ech**2),
                    kernel_initializer='glorot_uniform', 
                    name="Decomposition"))
    model.add(Flatten())

    model.add(Dense(128, activation='linear',
                    kernel_initializer='glorot_uniform', 
                    name="Fitt1"))

    model.add(Dense(32, activation='linear',
                    kernel_initializer='glorot_uniform',  
                    name="Fitt2"))

    model.add(Dense(32, activation='linear',
                    kernel_initializer='glorot_uniform',  
                    name="Fitt3"))

    model.add(Dense(32, activation='linear',
                    kernel_initializer='glorot_uniform',  
                    name="Fitt4"))

    model.add(Dense(32, activation='linear',
                    kernel_initializer='glorot_uniform',   
                    name="Fitt5"))

    model.add(Dense(8, activation='linear',
                    kernel_initializer='glorot_uniform', 
                    kernel_constraint=SequentialConstraint([
                        NonNeg(), 
                        NonNeg(), 
                        MinMaxNorm(min_value=allow, max_value=alhigh), 
                        MinMaxNorm(min_value=below, max_value=behigh),
                        MinMaxNorm(min_value=galow, max_value=gahigh),
                        MinMaxNorm(min_value=x0low, max_value=x0high),
                        MinMaxNorm(min_value=y0low, max_value=y0high),
                        MinMaxNorm(min_value=z0low, max_value=z0high)
                    ]), 
                    bias_constraint=SequentialConstraint([
                        NonNeg(), 
                        NonNeg(), 
                        MinMaxNorm(min_value=allow, max_value=alhigh), 
                        MinMaxNorm(min_value=below, max_value=behigh),
                        MinMaxNorm(min_value=galow, max_value=gahigh),
                        MinMaxNorm(min_value=x0low, max_value=x0high),
                        MinMaxNorm(min_value=y0low, max_value=y0high),
                        MinMaxNorm(min_value=z0low, max_value=z0high)
                    ]), 
                    name="Result1"))


    print(model.summary())
    # Compiler le modèle

    loss = tf.keras.losses.Huber(delta=0.5, reduction="auto", name="huber_loss")

    # ============= Optimiseur ================
    adam = Adam(lr=0.0079, 
                beta_1=0.81, 
                beta_2=0.95, 
                epsilon=1e-08, 
                decay=0.009)

    model.compile(optimizer=adam, 
                loss=loss, 
                metrics=['accuracy'])

    # Entraînement
    model.fit(X_train, y_train, epochs=epoch, 
            batch_size=batch, 
            validation_data=(X_test, y_test))

    # Validate the model's performance on a separate set of data
    loss, accuracy = model.evaluate(X_test, y_test)
    return model


# =========================================================================================
# =========================================================================================
# Obtentien les rayons
def get_rayons(nb_tot:int = -1, epoch:int = 250, batch:int = 32, select:float=0.9, parameters: np.ndarray = np.array([-10, 10, -10, 10, -10, 10, 5, 20, 0.1, 5, 0, 2*np.pi, 0, 2*np.pi, 0, 2*np.pi, "uniform", 0.1, -0.1, 0.5, False]), pat:int=3, Ech:int=-1):
    
    if(Ech == -1):
        print("Veuillez renseigner exactement le nombre d'achantillon.")
        exit()

    #========================== Soit le default array ================
    # x0low, x0high, y0low, y0high, z0low, z0high
    # r1low, r1high, r2low, r2high
    # allow, alhigh, below, behigh, galow, gahigh
    # mode, first_p, second_p
    # ampl
    # ================================================================

    x0low = parameters[0]
    x0high = parameters[1]

    y0low = parameters[2]
    y0high = parameters[3]

    z0low = parameters[4]
    z0high = parameters[5]

    r1low = parameters[6]
    r1high = parameters[7]

    r2low = parameters[8]
    r2high = parameters[9]

    allow = parameters[10] 
    alhigh = parameters[11]
    below = parameters[12] 
    behigh = parameters[13]
    galow = parameters[14] 
    gahigh = parameters[15]

    mode = parameters[16] 
    first_p = parameters[17] 
    second_p = parameters[18]
    ampl = parameters[19]
    save_fig = parameters[20]

    # ------------------------------------------------ 

    file_names = os.listdir("dataset/")
    file_names = sorted(file_names)
    n = int(len(file_names) * select)
    print(n)
    full_dataset = np.concatenate([np.load("dataset/"+f) for f in file_names[:n]])

    # Séparation du dataset en X (entrée) et Y (sortie)
    X = full_dataset[:,:3*Ech**2] # (x ,y ,z)
    print(X.shape)
    X = X.reshape((n, 3, Ech**2))
    y = full_dataset[:,3*Ech**2:3*Ech**2+2] # pour les autres paramètres

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
    # Dû à la version 1.15 de tensorflow (inapte à la 2.0)
    class SequentialConstraint(tf.keras.constraints.Constraint):
        def __init__(self, constraints):
            self.constraints = constraints

        def __call__(self, w):
            for constraint in self.constraints:
                w = constraint(w)
            return w

    model = Sequential()
    model.add(Dense(256, activation='relu', input_shape=(3,Ech**2)))
    model.add(Flatten())
    model.add(Dense(64, activation='linear',
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
    model.add(Dense(2, activation='linear'))


    print(model.summary())
    # Compiler le modèle

    loss = tf.keras.losses.Huber(delta=0.85, reduction="auto", name="huber_loss")

    # ============= Optimiseur ================
    adam = Adam(lr=0.0085, 
                beta_1=0.86, 
                beta_2=0.95, 
                epsilon=1e-08, 
                decay=0.009)

    model.compile(optimizer=adam, 
                loss=loss, 
                metrics=['accuracy'])

    # Entraînement
    model.fit(X_train, y_train, epochs=epoch, 
            batch_size=batch, 
            validation_data=(X_test, y_test))

    # Validate the model's performance on a separate set of data
    loss, accuracy = model.evaluate(X_test, y_test)
    model.save("modele/rayon_"+str(Ech**2)+'.h5')

    return model

# =========================================================================================
# =========================================================================================
# Obtentien les angles
def get_angles(nb_tot:int = -1, epoch:int = 250, batch:int = 32, select:float=0.9, parameters: np.ndarray = np.array([-10, 10, -10, 10, -10, 10, 5, 20, 0.1, 5, 0, 2*np.pi, 0, 2*np.pi, 0, 2*np.pi, "uniform", 0.1, -0.1, 0.5, False]), pat:int=3, Ech:int=-1):
    
    if(Ech == -1):
        print("Veuillez renseigner exactement le nombre d'achantillon.")
        exit()

    #========================== Soit le default array ================
    # x0low, x0high, y0low, y0high, z0low, z0high
    # r1low, r1high, r2low, r2high
    # allow, alhigh, below, behigh, galow, gahigh
    # mode, first_p, second_p
    # ampl
    # ================================================================

    x0low = parameters[0]
    x0high = parameters[1]

    y0low = parameters[2]
    y0high = parameters[3]

    z0low = parameters[4]
    z0high = parameters[5]

    r1low = parameters[6]
    r1high = parameters[7]

    r2low = parameters[8]
    r2high = parameters[9]

    allow = parameters[10] 
    alhigh = parameters[11]
    below = parameters[12] 
    behigh = parameters[13]
    galow = parameters[14] 
    gahigh = parameters[15]

    mode = parameters[16] 
    first_p = parameters[17] 
    second_p = parameters[18]
    ampl = parameters[19]
    save_fig = parameters[20]

    # ------------------------------------------------ 

    file_names = os.listdir("dataset/")
    file_names = sorted(file_names)
    n = int(len(file_names) * select)
    print(n)
    full_dataset = np.concatenate([np.load("dataset/"+f) for f in file_names[:n]])

    # Séparation du dataset en X (entrée) et Y (sortie)
    X = full_dataset[:,:3*Ech**2] # (x ,y ,z)
    print(X.shape)
    X = X.reshape((n, 3, Ech**2))
    y = full_dataset[:,3*Ech**2+2:3*Ech**2+2+3] # pour les autres paramètres

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
    # Dû à la version 1.15 de tensorflow (inapte à la 2.0)
    class SequentialConstraint(tf.keras.constraints.Constraint):
        def __init__(self, constraints):
            self.constraints = constraints

        def __call__(self, w):
            for constraint in self.constraints:
                w = constraint(w)
            return w

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
    model.add(Dense(32, activation='linear',
                    kernel_initializer='uniform',
                    bias_initializer='zeros'))
    model.add(Dense(16, activation='linear',
                    kernel_initializer='uniform',
                    bias_initializer='zeros'))
    model.add(Dense(3, activation='linear'))


    print(model.summary())
    # Compiler le modèle

    loss = tf.keras.losses.Huber(delta=0.15, reduction="auto", name="huber_loss")

    # ============= Optimiseur ================
    adam = Adam(lr=0.0075, 
                beta_1=0.81, 
                beta_2=0.95, 
                epsilon=1e-08, 
                decay=0.004)

    model.compile(optimizer=adam, 
                loss=loss, 
                metrics=['accuracy'])

    # Entraînement
    model.fit(X_train, y_train, epochs=epoch, 
            batch_size=batch, 
            validation_data=(X_test, y_test))

    # Validate the model's performance on a separate set of data
    loss, accuracy = model.evaluate(X_test, y_test)
    model.save("modele/angles_"+str(Ech**2)+'.h5')

    return model

# =========================================================================================
# =========================================================================================
# Obtention les offsets
def get_offsets(nb_tot:int = -1, epoch:int = 250, batch:int = 32, select:float=0.9, parameters: np.ndarray = np.array([-10, 10, -10, 10, -10, 10, 5, 20, 0.1, 5, 0, 2*np.pi, 0, 2*np.pi, 0, 2*np.pi, "uniform", 0.1, -0.1, 0.5, False]), pat:int=3, Ech:int=-1):
    
    if(Ech == -1):
        print("Veuillez renseigner exactement le nombre d'achantillon.")
        exit()

    #========================== Soit le default array ================
    # x0low, x0high, y0low, y0high, z0low, z0high
    # r1low, r1high, r2low, r2high
    # allow, alhigh, below, behigh, galow, gahigh
    # mode, first_p, second_p
    # ampl
    # ================================================================

    x0low = parameters[0]
    x0high = parameters[1]

    y0low = parameters[2]
    y0high = parameters[3]

    z0low = parameters[4]
    z0high = parameters[5]

    r1low = parameters[6]
    r1high = parameters[7]

    r2low = parameters[8]
    r2high = parameters[9]

    allow = parameters[10] 
    alhigh = parameters[11]
    below = parameters[12] 
    behigh = parameters[13]
    galow = parameters[14] 
    gahigh = parameters[15]

    mode = parameters[16] 
    first_p = parameters[17] 
    second_p = parameters[18]
    ampl = parameters[19]
    save_fig = parameters[20]

    # ------------------------------------------------ 

    file_names = os.listdir("dataset/")
    file_names = sorted(file_names)
    n = int(len(file_names) * select)
    print(n)
    full_dataset = np.concatenate([np.load("dataset/"+f) for f in file_names[:n]])

    # Séparation du dataset en X (entrée) et Y (sortie)
    X = full_dataset[:,:3*Ech**2] # (x ,y ,z)
    print(X.shape)
    X = X.reshape((n, 3, Ech**2))
    y = full_dataset[:,3*Ech**2+2+3:3*Ech**2+2+3+3] # pour les autres paramètres

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
    # Dû à la version 1.15 de tensorflow (inapte à la 2.0)
    class SequentialConstraint(tf.keras.constraints.Constraint):
        def __init__(self, constraints):
            self.constraints = constraints

        def __call__(self, w):
            for constraint in self.constraints:
                w = constraint(w)
            return w

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
    model.add(Dense(32, activation='linear',
                    kernel_initializer='uniform',
                    bias_initializer='zeros'))
    model.add(Dense(32, activation='linear',
                    kernel_initializer='uniform',
                    bias_initializer='zeros'))
    model.add(Dense(16, activation='linear',
                    kernel_initializer='uniform',
                    bias_initializer='zeros'))
    model.add(Dense(32, activation='linear',
                    kernel_initializer='uniform',
                    bias_initializer='zeros'))
    model.add(Dense(3, activation='linear'))


    print(model.summary())
    # Compiler le modèle

    loss = tf.keras.losses.Huber(delta=0.65, reduction="auto", name="huber_loss")

    # ============= Optimiseur ================
    adam = Adam(lr=0.0092, 
                beta_1=0.81, 
                beta_2=0.95, 
                epsilon=1e-08, 
                decay=0.0105)

    model.compile(optimizer=adam, 
                loss=loss, 
                metrics=['accuracy'])

    # Entraînement
    model.fit(X_train, y_train, epochs=epoch, 
            batch_size=batch, 
            validation_data=(X_test, y_test))

    # Validate the model's performance on a separate set of data
    loss, accuracy = model.evaluate(X_test, y_test)
    model.save("modele/offsets_"+str(Ech**2)+'.h5')

    return model