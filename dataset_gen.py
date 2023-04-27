from tore_gen import tore
import numpy as np

def generate_dataset(nb: int = -1, i: int = 0, save: bool = False, path: str = "dataset/torus_datasetN", Ech: int = 50, parameters: np.ndarray = np.array([-10, 10, -10, 10, -10, 10, 5, 20, 0.1, 5, 0, 2*np.pi, 0, 2*np.pi, 0, 2*np.pi, "uniform", 0.1, -0.1, 0.5, False]), show:bool = True, info:bool = False):
    #========================== Soit le default array ================
    # x0low, x0high, y0low, y0high, z0low, z0high
    # r1low, r1high, r2low, r2high
    # allow, alhigh, below, behigh, galow, gahigh
    # mode, first_p, second_p
    # ampl
    # ================================================================

    x0low = float(parameters[0])
    x0high = float(parameters[1])

    y0low = float(parameters[2])
    y0high = float(parameters[3])

    z0low = float(parameters[4])
    z0high = float(parameters[5])

    r1low = float(parameters[6])
    r1high =float(parameters[7])

    r2low = float(parameters[8])
    r2high = float(parameters[9])

    allow = float(parameters[10])
    alhigh = float(parameters[11])
    below = float(parameters[12])
    behigh = float(parameters[13])
    galow = float(parameters[14])
    gahigh = float(parameters[15])

    mode = str(parameters[16])
    first_p = float(parameters[17])
    second_p = float(parameters[18])
    ampl = float(parameters[19])
    save_fig = bool(parameters[20])

    # ------------------------------------------------ 

    if(nb != -1):
        # génération des paramètres
        r1 = np.random.uniform(low=r1low, high=r1high)
        r2 = np.random.uniform(low=r2low, high=r2high)
        alpha = np.random.uniform(low=allow, high=alhigh)
        beta = np.random.uniform(low=below, high=behigh)
        gamma = np.random.uniform(low=galow, high=gahigh)
        x0 = np.random.uniform(low=x0low, high=x0high)
        y0 = np.random.uniform(low=y0low, high=y0high)
        z0 = np.random.uniform(low=z0low, high=z0high)

        # Attention au nb

        X = tore(x0=x0, y0=y0, z0=z0, r1=r1, r2=r2, alpha=alpha, beta=beta, gamma=gamma, nbp=Ech, mode=mode, first_p=first_p, second_p=second_p, show=show, ampl=ampl, save=save_fig, nom="TorusN"+str(i))
        Y = np.array([r1, r2, alpha, beta, gamma, x0, y0, z0])

        X = X.reshape(1,-1)
        Y = Y.reshape(1,-1)

        print(X.shape)
        print(Y.shape)
        print(str(i+1)+"/"+str(nb))

        np.save(path+str(i)+'.npy', np.hstack((X, Y)))

        param = {"r1":r1, "r2":r2, "alpha":alpha, "beta":beta, "gamma":gamma, "x0":x0, "y0":y0, "z0":z0}

        if(info == False):
            print("Pour la génération :"+str(i))
            for key, value in param.items():
                print("{:<10} : {:<10}".format(key, value))

    else:
        pass
    