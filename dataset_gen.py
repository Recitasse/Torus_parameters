from tore_gen import tore
import numpy as np

def generate_dataset(nb: int = -1, i: int = 0, save: bool = False, path: str = "dataset/torus_datasetN", Ech: int = 50, parameters: np.ndarray = np.array([-10, 10, -10, 10, -10, 10, 5, 20, 0.1, 5, 0, 2*np.pi, 0, 2*np.pi, 0, 2*np.pi, False]), show:bool = True):
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

        print("Pour la génération :"+str(i))
        for key, value in param.items():
            print("{:<10} : {:<10}".format(key, value))

    else:
        pass
    