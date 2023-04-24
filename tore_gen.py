import numpy as np
import random as r
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

def tore(x0: float = 0.0, y0: float = 0.0, z0: float = 0.0, r1:float = 1.0, r2: float = 0.2, alpha: float = 0.0, beta: float = 0.0, gamma: float = 0.0, nbp: int = 50, mode: str = "none", first_p: float = 0.0, second_p: float = 1.0, show: bool = False, ampl: float = 0, save: bool = False, nom: str = "dataset/"):
    r.seed(42)
    
    def Rot(a, b, c):
        # Dans la notations Eulériennes z-x-z
        def Rx(a):
            return np.array([[1, 0, 0], [0, np.cos(a), np.sin(a)], [0, -np.sin(a), np.cos(a)]])
        def Ry(b):
            return np.array([[np.cos(b), 0, -np.sin(b)], [0, 1, 0], [np.sin(b), 0, np.cos(b)]])
        def Rz(c):
            return np.array([[np.cos(c), np.sin(c), 0], [-np.sin(c), np.cos(c), 0], [0, 0, 1]])
        
        tempo = np.dot(Rz(a), Rx(b))
        return np.dot(tempo, Rz(c))

    # ====== Les distributions aléatoires

    def uni(a, b, n):
        Noise = np.zeros((3, n))
        Noiser = np.zeros((3, n))
        for j in range(3):
            for i in range(n):
                Noise[j,i] = r.uniform(a, b)
                Noiser[j,i] = r.uniform(a, b)
        return Noise, Noiser

    def normal(a, b, n):
        Noiser = np.zeros((3, n))
        Noise = np.zeros((3, n))
        for j in range(3):
            for i in range(n):
                Noise[j,i] = r.normalvariate(a, b)
                Noiser[j,i] = r.normalvariate(a, b)
        return Noise, Noiser
    
    def expo(b, n):
        Noiser = np.zeros((3, n))
        Noise = np.zeros((3, n))
        for j in range(3):
            for i in range(n):
                Noise[j,i] = r.expovariate(b)
                Noiser[j,i] = r.expovariate(b)
        return Noise, Noiser

    # ===================================

    case = mode
    if(case == "none"):
            Noise = np.zeros((3, nbp**2))
            Noiser = np.zeros((3, nbp))
    elif(case == "normal"):
            Noise, Noiser = normal(first_p, second_p, nbp**2)
    elif(case == "expo"):
            Noise, Noiser = expo(second_p, nbp**2)
    elif(case == "uniform"):
            Noise, Noiser = uni(first_p, second_p, nbp**2)
    else:
            Noise = np.zeros((3, nbp**2))
            Noiser = np.zeros((3, nbp))

    # ============ générer les points en 3D ==========

    X = np.zeros((3,nbp**2))
    t1 = np.linspace(0,2*np.pi,nbp)
    t2 = np.linspace(0,2*np.pi,nbp)

    k = 0
    for i in range(nbp):
        for j in range(nbp):

            # Ce ne sont pas des variables aléatoires pour rajouter du bruit, mais pour créer une homogénéité de la génération
            b1 = np.random.uniform(0,np.pi)
            b2 = np.random.uniform(-np.pi,np.pi)

            X[0, k] = (Noiser[0,j]*ampl+r1+r2*np.cos(t1[i]+b1))*np.cos(t2[j]+b2) + Noise[0, j]
            X[1, k] = (Noiser[1,j]*ampl+r1+r2*np.cos(t1[i]+b1))*np.sin(t2[j]+b2) + Noise[1, j]
            X[2, k] = (Noiser[2,j]*ampl+r2)*np.sin(t1[i]+b1) + Noise[2, j]
            X[:, k] = np.dot(Rot(alpha, beta, gamma),X[:, k])+[x0, y0, z0]
            k += 1

    if(show == True):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X[0, :], X[1, :], X[2, :], marker='o')

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Torus')
        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(-3.5, 3.5)
        ax.set_zlim(-3.5, 3.5)
        if(save == True):
            fig.savefig("datasetIm/"+nom+".png")
            plt.close(fig)

    return X

def tore_view(x0: float = 0.0, y0: float = 0.0, z0: float = 0.0, r1:float = 1.0, r2: float = 0.5, alpha: float = 0.0, beta: float = 0.0, gamma: float = 0.0, nbp: int = 50):
    
    def Rot(a, b, c):
        # Dans la notations Eulériennes z-x-z
        def Rx(a):
            return np.array([[1, 0, 0], [0, np.cos(a), np.sin(a)], [0, -np.sin(a), np.cos(a)]])
        def Ry(b):
            return np.array([[np.cos(b), 0, -np.sin(b)], [0, 1, 0], [np.sin(b), 0, np.cos(b)]])
        def Rz(c):
            return np.array([[np.cos(c), np.sin(c), 0], [-np.sin(c), np.cos(c), 0], [0, 0, 1]])
        
        tempo = np.dot(Rz(a), Rx(b))
        return np.dot(tempo, Rz(c))
    
    X = np.zeros((3,nbp**2))

    t1 = np.linspace(0,2*np.pi,nbp)
    t2 = np.linspace(0,2*np.pi,nbp)

    k = 0
    for i in range(nbp):
        for j in range(nbp):

            # Ce ne sont pas des variables aléatoires pour rajouter du bruit, mais pour créer une homogénéité de la génération
            b1 = np.random.uniform(0,np.pi)
            b2 = np.random.uniform(-np.pi,np.pi)

            X[0, k] = (r1+r2*np.cos(t1[i]+b1))*np.cos(t2[j]+b2)
            X[1, k] = (r1+r2*np.cos(t1[i]+b1))*np.sin(t2[j]+b2)
            X[2, k] = r2*np.sin(t1[i]+b1)
            X[:, k] = np.dot(Rot(alpha, beta, gamma),X[:, k])+[x0, y0, z0]
            k += 1

    return X