""" Code adapted from: https://github.com/simonesilvetti/pyCheck
""" 

import os
import time
import numpy as np
from math import erf
import pickle5 as pickle
from scipy.stats import norm
from scipy.optimize import minimize
from evaluation_metrics import execution_time, intervals_intersection
from sklearn.gaussian_process.kernels import RBF


class EPupdate():
    def __init__(self):
        self.TermNew = None
        self.logZterms = None
        self.logZ = None


class CavGauss():
    def __init__(self):
        self.diagV = None
        self.m = None

class Gauss():
    def __init__(self):
        self.LikPar_p = None
        self.xGauss = None
        self.wGauss = None
        self.logwGauss = None
        self.C = []
        self.LC = None
        self.LC_t = None
        self.L = None
        self.W = None
        self.diagV = None
        self.m = None
        self.logZloo = None
        self.logZappx = None
        self.logZterms = None
        self.logZ = None
        self.Term = None

class smMC_GPEP(object):

    def __init__(self):
        self.CORRECTION_FACTOR = 1
        self.CORRECTION = 1E-4
        self.eps_damp = 0.5

    def transform_data(self, data):
        x = data["params"]
        p = data["labels"]
        n_samples, n_trials = p.shape
        y = np.sum(p,axis=1)/n_trials
        y = np.expand_dims(y,axis=1)
        return x, y, n_samples, n_trials

    def getProbability(self, mean, variance):
        return norm.cdf(mean / np.sqrt(1 + variance))

    def getBounds(self, mean, variance, z, x_train):
        quantiles_interval = [mean - z * np.sqrt(variance), mean + z * np.sqrt(variance)]
        # bounds = norm.cdf(np.tile(1 / np.sqrt(1 + variance), (x_train.shape[1], 1)) * quantiles_interval)
        bounds = norm.cdf(np.tile(1 / np.sqrt(1 + variance), (2, 1)) * quantiles_interval)
        return bounds[0, :], bounds[1, :]

    def make_predictions(self, x_train, x_test, z=1.96):
        mean, variance = self.latentPrediction(x_train=x_train, x_test=x_test)
        q1, q2 = self.getBounds(mean=mean, variance=variance, z=z, x_train=x_train)
        post_mean = self.getProbability(mean, variance)
        return post_mean, q1, q2

    def fit(self, x_train, y_train, n_trajectories):
        start = time.time()
        aa, bb = self.getDefaultHyperarametersRBF(x_train, y_train)
        objectivefunctionWrap = lambda x: self.objectivefunction(x_train, y_train, n_trajectories=n_trajectories, l=x)

        res = minimize(objectivefunctionWrap, bb, method='L-BFGS-B', bounds=((0.5 * bb, 2 * bb),))
        r = RBF(res.x)
        self.kernel = r
        invC, mu_tilde, sigma_tilde = self.doTraining(x_train, y_train, n_trajectories=n_trajectories)
        self.invC=invC
        self.mu_tilde=mu_tilde
        self.sigma_tilde=sigma_tilde

        training_time = execution_time(start=start, end=time.time())
        print("\nTraining time =", training_time)
        self.training_time = training_time

    def doTraining(self, x_train, y_train, n_trajectories):
        gauss = self.expectationPropagation(x_train, y_train, n_trajectories=n_trajectories, tolerance=1e-6)
        v_tilde = gauss.Term[:, 0]
        tau_tilde = gauss.Term[:, 1]
        diag_sigma_tilde = 1 / tau_tilde
        mu_tilde = v_tilde * diag_sigma_tilde
        sigma_tilde = np.diag(diag_sigma_tilde)
        invC = np.linalg.solve(gauss.C + sigma_tilde, np.eye(len(mu_tilde)))
        return invC, mu_tilde, sigma_tilde

    def computeMarginalMoments(self, gauss, Term, logdet_LC):
        # // (repmat(Term(:,2),1,N).*Gauss.LC)
        N = len(Term)
        tmp = np.tile(Term[:, 1], (N, 1)).T * gauss.LC
        A = np.matrix.dot(gauss.LC_t, tmp) + np.eye(N) * self.CORRECTION_FACTOR
        # // Serious numerical stability issue with the calculation
        # // of A (i.e. A = LC' * tmp + I)
        # // as it does not appear to be PD for large amplitudes
        gauss.L = np.linalg.cholesky(A)
        # // Gauss.W = Gauss.L\(Gauss.LC');
        gauss.W = np.linalg.solve(gauss.L, gauss.LC_t)
        # // Gauss.diagV = sum(Gauss.W.*Gauss.W,1)';
        tmp = gauss.W * gauss.W
        # gauss.diagV = np.zeros(shape=(N, 1))
        # for (int i = 0; i < N; i++)
        # gauss.diagV.put(i, tmp.getColumn(i).sum());
        gauss.diagV = np.array([np.sum(tmp, 0)]).T
        # // or
        # // gauss.diagV = gauss.W.transpose().mmul(gauss.W).diag();
        #
        # // Gauss.m = Gauss.W'*(Gauss.W*Term(:,1));
        tmp = np.dot(gauss.W, Term[:, 0])
        gauss.m = np.array([np.dot(gauss.W.T, tmp)]).T
        # // logdet = -2*sum(log(diag(Gauss.L))) + 2*sum(log(diag(Gauss.LC)));
        logdet = 0
        sum = 0
        tmp = np.diag(gauss.L)
        # for (int i = 0; i < tmp.getLength(); i++)
        # sum += Math.log(tmp.get(i));
        # logdet += -2 * sum;
        logdet += -2.0 * np.sum(np.log(tmp))
        # // sum = 0;
        # // tmp = gauss.LC.diag();
        # // for (int i = 0; i < tmp.getLength(); i++)
        # // sum += Math.log(tmp.get(i));

        logdet += logdet_LC

        # // logZappx = 0.5*(Gauss.m'*Term(:,1) + logdet);
        logZappx = 0.5 * (np.dot(gauss.m.transpose(), Term[:, 0]) + logdet)
        return logZappx, gauss

    def gausshermite(self, n, x, w):
        x0 = np.zeros(shape=(len(x), 1))
        w0 = np.zeros(shape=(len(w), 1))
        m = int((n + 1) / 2)
        z = 0
        pp = 0
        p1 = 0
        p2 = 0
        p3 = 0
        for i in range(0, m):
            if (i == 0):
                z = np.sqrt(2 * n + 1) - 1.85575 * (2 * n + 1) ** (-0.16667)
            elif (i == 1):
                z = z - 1.14 * n ** 0.426 / z
            elif (i == 2):
                z = 1.86 * z - 0.86 * x0[0]
            elif (i == 3):
                z = 1.91 * z - 0.91 * x0[1]
            else:
                z = 2.0 * z - x0[i - 2]

            for its in range(0, 10):
                p1 = 1 / np.sqrt(np.sqrt(np.pi))
                p2 = 0
                for j in range(1, n + 1):
                    p3 = p2
                    p2 = p1
                    a = z * np.sqrt(2 / j) * p2
                    b = np.sqrt((j - 1) / j) * p3
                    p1 = a - b

                pp = np.sqrt(2 * n) * p2
                z1 = z
                z = z1 - p1 / pp
                if (np.abs(z - z1) < 2.2204e-16):
                    break

            x0[i] = z
            x0[n - 1 - i] = -z
            w0[i] = 2 / (pp * pp)
            w0[n - 1 - i] = w0[i]

        w0 = w0 / np.sqrt(np.pi)
        x0 = x0 * np.sqrt(2)
        x0 = np.sort(x0)[::-1]
        x = x0
        w = w0
        return x, w

    def computeCavities(self, gauss, Term):
        cavGauss = CavGauss()
        # // C = Gauss.diagV;
        C = gauss.diagV
        # // s = 1./(1 + Term(:,2).*C)
        appo = np.array([a * b for a, b in zip(Term[:, 1], C)])
        s = np.ones(shape=(len(C), 1)) / (appo + 1)
        # // CavGauss.diagV = s. * C;
        cavGauss.diagV = s * C
        # // CavGauss.m = s. * (Gauss.m + Term(:, 1).*C);
        appo = np.array([a * b for a, b in zip(Term[:, 0], C)])
        cavGauss.m = s * (gauss.m + appo)
        return cavGauss

    def ep_update(self, cavGauss, Term, LikPar_p, LikPar_q, xGauss, wGauss):
        update = EPupdate()
        Cumul = np.zeros(shape=(len(LikPar_p), 2))
        logZ, Cumul = self.GaussHermiteNQ(LikPar_p, LikPar_q, cavGauss.m, cavGauss.diagV, xGauss, wGauss, Cumul)
        update.logZ = np.array([logZ]).T
        m2 = cavGauss.m * cavGauss.m
        logV = np.log(cavGauss.diagV)
        cumul1 = np.array([Cumul[:, 0] * Cumul[:, 0]]).T
        cumul2 = np.log(np.array([Cumul[:, 1]]).T)
        tmp = m2 / (cavGauss.diagV) + logV - (cumul1 / np.array([Cumul[:, 1]]).T + cumul2)
        update.logZterms = update.logZ + tmp * 0.5
        ones = np.ones(shape=(len(LikPar_p), 1))
        TermNew = np.zeros(shape=(len(LikPar_p), 2))
        c1 = np.array([Cumul[:, 0] / Cumul[:, 1]]).T - cavGauss.m / cavGauss.diagV
        c2 = ones / np.array([Cumul[:, 1]]).T - ones / cavGauss.diagV
        TermNew[:, 0] = c1[:, 0]
        TermNew[:, 1] = c2[:, 0]
        TermNew = (1 - self.eps_damp) * Term + self.eps_damp * TermNew
        update.TermNew = TermNew
        return update
    
    def GaussHermiteNQ(self, FuncPar_p, FuncPar_q, m, v, xGH, logwGH, Cumul):
        stdv = np.sqrt(v)
        Nnodes = len(xGH)
        tmp = np.dot(stdv, xGH.transpose())
        Y = tmp + np.tile(m, (1, Nnodes))
        tmp = self.logprobitpow(Y, FuncPar_p, FuncPar_q)
        G = tmp + np.tile(logwGH.transpose(), (len(tmp), 1))
        maxG = np.max(G, 1)
        G = G - np.tile(maxG, (Nnodes, 1)).T
        expG = np.exp(G)
        denominator = np.sum(expG, 1)
        logZ = maxG + np.log(denominator)
        deltam = stdv * (np.dot(expG, xGH)) / np.array([denominator]).T
        appo = m + deltam
        Cumul[:, 0] = appo[:, 0]
        appo = v * np.dot(expG, xGH ** 2) / np.array([denominator]).T - deltam ** 2
        Cumul[:, 1] = appo[:, 0]
        return logZ, Cumul
    
    def logprobitpow(self, X, LikPar_p, LikPar_q):
        n = X.shape[0]
        m = X.shape[1]
        Y = np.zeros(shape=(n, m))
        for i in range(0, n):
            for j in range(0, m):
                Y[i][j] = self.ncdflogbc(X[i][j])
        Za = Y * np.tile(LikPar_p, (1, m))
        Y = np.zeros(shape=(n, m))
        for i in range(0, n):
            for j in range(0, m):
                Y[i][j] = self.ncdflogbc(-X[i][j])
        Zb = Y * np.tile(LikPar_q, (1, m))
        return Za + Zb


    def ncdflogbc(self, x):
        sqrt2 = np.sqrt(2)
        invSqrt2 = 1 / sqrt2
        log2 = np.log(2)
        treshold = -sqrt2 * 5
        z = -x
        if (x >= 0):
            return np.log(1 + erf(x * invSqrt2)) - log2
        if (treshold < x):
            return np.log(1 - erf(-x * invSqrt2)) - log2
        return -0.5 * np.log(np.pi) - log2 - 0.5 * z * z - np.log(z) + np.log(
            1 - 1 / z + 3 / z ** 4 - 15 / z ** 6 + 105 / z ** 8 - 945 / z ** 10)

    def expectationPropagation(self, x_train, y_train, n_trajectories, tolerance, max_iters=1000):

        print("\nEP:", end="")
        gauss = Gauss()
        p = self.kernel(x_train)
        gauss.C = p

        gauss.C = gauss.C + self.CORRECTION * np.eye(len(gauss.C))
        gauss.LC = np.linalg.cholesky(gauss.C)
        gauss.LC_t = gauss.LC.transpose()
        gauss_LC_diag = np.diag(gauss.LC)
        logdet_LC = 2 * np.sum(np.log(gauss_LC_diag))
        logZprior = 0.5 * logdet_LC
        n = len(x_train)
        logZterms = np.zeros(shape=(n, 1))
        logZloo = np.zeros(shape=(n, 1))
        Term = np.zeros(shape=(n, 2))
        appo, gauss = self.computeMarginalMoments(gauss, Term, logdet_LC)

        # Stuff related to the likelihood
        gauss.LikPar_p = y_train * n_trajectories
        gauss.LikPar_q = np.ones(shape=(n, 1)) * n_trajectories - gauss.LikPar_p
        NODES = 96
        gauss.xGauss = np.zeros(shape=(NODES, 1))
        gauss.wGauss = np.zeros(shape=(NODES, 1))
        gauss.xGauss, gauss.wGauss = self.gausshermite(NODES, gauss.xGauss, gauss.wGauss)
        gauss.logwGauss = np.log(gauss.wGauss)

        # initialize cycle control
        tol = tolerance
        logZold = 0
        logZ = 2 * tol
        steps = 0
        logZappx = 0
        while ((np.abs(logZ - logZold) > tol) & (steps < max_iters)):
            print("...", end="")
            # cycle control
            steps = steps + 1
            logZold = logZ
            cavGauss = self.computeCavities(gauss, -Term)

            update = self.ep_update(cavGauss, Term, gauss.LikPar_p, gauss.LikPar_q, gauss.xGauss, gauss.logwGauss)
            Term = update.TermNew
            logZterms = update.logZterms
            logZloo = update.logZ

            logZappx, gauss = self.computeMarginalMoments(gauss, Term, logdet_LC)
            logZ = logZterms.sum() + logZappx

        # finishing
        logZ = logZ - logZprior
        gauss.logZloo = np.sum(logZloo)
        gauss.logZappx = logZappx
        gauss.logZterms = logZterms
        gauss.logZ = logZ
        gauss.Term = Term
        return gauss

    def getDefaultHyperarametersRBF(self, x_train, y_train):
        signal = 0.5 * (np.max(y_train) - np.min(x_train))
        sum = 0
        n, dim = x_train.shape
        for d in range(0, dim):
            max = -float('inf')
            min = float('inf')
            for i in range(0, n):
                curr = x_train[i][d]
                if (curr > max):
                    max = curr
                if (curr < min):
                    min = curr
            sum += (max - min) / 10.0
        lengthScale = sum / dim
        return signal, lengthScale

    def objectivefunction(self, x_train, y_train, n_trajectories, l):
        r = RBF(l)
        self.kernel = r
        return self.getMarginalLikelihood(x_train, y_train, n_trajectories)

    def getMarginalLikelihood(self, x_train, y_train, n_trajectories):
        gauss = self.expectationPropagation(x_train, y_train, n_trajectories, tolerance=1e-3)
        return gauss.logZ

    def latentPrediction(self, x_train, x_test): # Rasmussen GPs p. 74
        kss = np.diag(self.kernel(x_test))
        ks = self.kernel(x_test, x_train)
        tmp = np.dot(ks, self.invC)
        fs = np.dot(tmp, self.mu_tilde)
        vfs = kss - (np.diag(np.dot(tmp, ks.transpose())))
        return fs, vfs

    def save(self, filepath, filename):

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        smc_dict = {"kernel":self.kernel, "invC":self.invC, "mu_tilde":self.mu_tilde}

        with open(os.path.join(filepath, filename+".pickle"), 'wb') as handle:
            pickle.dump(smc_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        file = open(os.path.join(filepath, filename+"_training_time.txt"),"w")
        file.writelines(self.training_time)
        file.close()

    def load(self, filepath, filename):

        with open(os.path.join(filepath, filename+".pickle"), 'rb') as handle:
            smc_dict = pickle.load(handle)

        # smc = smc_dict["smc"]
        self.kernel = smc_dict["kernel"]
        self.invC = smc_dict["invC"]
        self.mu_tilde = smc_dict["mu_tilde"]

        file = open(os.path.join(filepath, filename+"_training_time.txt"),"r+")
        training_time = file.read()
        print(f"\nTraining time = {training_time}")
        return training_time

    def eval_gp(self, x_train, x_val, y_val, n_samples, n_trials, z=1.96):

        start = time.time()

        if y_val.shape != (n_samples, n_trials):
            raise ValueError("y_val should be bernoulli trials")

        satisfaction_prob = y_val.mean(1).flatten()
        assert satisfaction_prob.min()>=0
        assert satisfaction_prob.max()<=1

        post_mean, q1, q2 = self.make_predictions(x_train=x_train, x_test=x_val)
        assert satisfaction_prob.shape == post_mean.shape
        assert satisfaction_prob.shape == q1.shape

        sample_variance = [((param_y-param_y.mean())**2).mean() for param_y in y_val]
        val_std = np.sqrt(sample_variance).flatten()
        validation_ci = (satisfaction_prob-(z*val_std)/np.sqrt(n_trials),satisfaction_prob+(z*val_std)/np.sqrt(n_trials))
        
        q1[q1<10e-6] = 0
        estimated_ci = (q1,q2)
        non_empty_intersections = np.sum(intervals_intersection(validation_ci,estimated_ci)>0)
        val_accuracy = 100*non_empty_intersections/n_samples
        assert val_accuracy <= 100

        val_dist = np.abs(satisfaction_prob-post_mean)
        mse = np.mean(val_dist**2)
        ci_uncertainty_area = q2-q1
        avg_uncertainty_area = np.mean(ci_uncertainty_area)

        print(f"Mean squared error: {mse}")
        print(f"Validation accuracy: {val_accuracy} %")
        print(f"Average uncertainty area:  {avg_uncertainty_area}\n")
        evaluation_dict = {"val_accuracy":val_accuracy, "mse":mse, 
            "uncertainty_area":ci_uncertainty_area, "avg_uncertainty_area":avg_uncertainty_area}

        evaluation_time = execution_time(start=start, end=time.time())
        print(f"Evaluation time = {evaluation_time}")
        evaluation_dict.update({"evaluation_time":evaluation_time})

        return post_mean, q1, q2, evaluation_dict

