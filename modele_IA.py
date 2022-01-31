import numpy as np

# Fonction modélisant le modèle linéaire F
def model (X, theta ):
    return X.dot( theta )

# Fonction de cout
def fonction_cout(X, Y, theta):
    m = len(X)
    return (1/(2*m))* np.sum(np.power((model(X, theta) - Y), 2))

# Fonction gradien
def gradient(X, Y, theta):
    m = len(X)
    return (1/m) * X.T.dot((model(X, theta) - Y))

# Fonction descente de gradient
def descente_gradient(X, Y, theta, alpha=1e-3, n_iterations=30):
    '''
    X : matrice de variable descriptive
    Y : matrice target
    theta : matrice contenant les coefficients
    alpha : learning rate
    n_iterations : nombre d'itération
    '''
    F_plot = [] # Variable permetant de stocker tout les modèles pour les représenter facilement
    J = [] # Variable pour stocker le coût
    
    for i in range(n_iterations):
        theta = theta - alpha*gradient(X, Y, theta) # Re-estimation de thêta
        F_plot.append(model(X, theta)) # Calcul du model avec le nouveau thêta
        J.append(fonction_cout(X, Y, theta)) # Calcul du coût du modèle
    
    return J, theta, F_plot