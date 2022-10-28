def LSM(T, r, sigma, K, S0, N, M, k, right=1, seed=1234):
    """
    Longstaff-schwartz method: Gustafsson Implementation
    
    T: Expiration Time 
    r: interest rate
    sigma: underlying vol
    K: strike
    S0: inital underlying
    N: # timesteps from t=0 to t=T
    M: # realizations (even)
    k: basis functions
    seed: random seed for reproducibility
    """

    np.random.seed(seed)
    t = np.arange(0, T, T / N)
    z = np.random.normal(size=(math.floor(M / 2), 1))
    w = (r - (sigma ** 2 / 2)) * T + sigma * np.sqrt(T) * np.vstack([z, -z])
    S = S0 * np.exp(w)
    sup_eb = {}
    inf_eb = {}

    if right == 1:  # put payoffs
        P = np.maximum.reduce([K - S, np.zeros(S.shape)]).reshape(1, -1)[0]
    elif right == 0:  # call payoffs
        P = np.maximum.reduce([S - K, np.zeros(S.shape)]).reshape(1, -1)[0]
    else:
        return

    for i in range(N - 2, 0, -1):
        z = np.random.normal(size=(math.floor(M / 2), 1))
        w = (t[i] * w) / t[i + 1] + sigma * np.sqrt(
            (T / N) * t[i] / t[i + 1]
        ) * np.vstack(
            [z, -z]
        )  # brownian bridge sampling
        S = S0 * np.exp(w)
        if right == 1:
            index = np.where(K > S)[0]
        elif right == 0:
            index = np.where(S < K)[0]
        X = S[index].reshape(1, -1)[0]  # prices and payoffs itm
        Y = P[index] * np.exp(-r * T / N).reshape(1, -1)[0]
        A = laguerre_basis(X, k)  # laguerre basis functions
        x, resid, rank, s = np.linalg.lstsq(A, Y, rcond=None)
        C = A.dot(x.reshape(1, -1)[0])  #  estimated continuation using ols

        if right == 1:
            E = K - X
        elif right == 0:
            E = X - K

        exP = np.where(C <= E)[0]  # indices better to exercise
        index_exP = index[exP]
        if len(exP) > 0:
            # sup_price=X[np.where(X[exP]==np.max(X[exP]))[0]]
            sup_price = np.mean(X[exP])
            sup_eb[i] = sup_price

        non_exP = np.where(C >= E)[0]
        if len(non_exP) > 0:
            # inf_price=X[np.where(X[non_exP]==np.min(X[non_exP]))[0]]
            inf_price = np.mean(X[non_exP])
            inf_eb[i] = inf_price

        rest = np.setdiff1d(
            np.arange(0, M), index_exP
        )  # new realizations optimal to exercise

        P[index_exP] = E[exP]  # update payoffs
        P[rest] = P[rest] * np.exp(-r * T / N)

    u = np.mean(P * np.exp(-r * T / N))

    return u, sup_eb, inf_eb
