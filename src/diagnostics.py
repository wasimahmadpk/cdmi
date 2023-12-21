import numpy as np
import pandas as pd
import torch
import seaborn as sns
from torch.autograd import Function
from matplotlib.patches import Patch
from DeepKnockoffs.mmd import mix_rbf_mmd2
import torch_two_sample.statistics_diff as diff_tests
import torch_two_sample.statistics_nondiff as nondiff_tests
import matplotlib.pyplot as plt
import sys

def PlotScatterHelper(A, B, ax=None):
    """
    Plot the entries of a matrix A vs those of B
    :param A: n-by-p data matrix
    :param B: n-by-p data matrix
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None

    ax.set_xlim([-0.2, 1.1])
    ax.set_ylim([-0.2, 1.1])

    for i in range(0,A.shape[0]-1):
        ax.scatter(A.diagonal(i),B.diagonal(i), alpha=0.75)

    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]

    for i in range(0,A.shape[0]-1):
        print("Matrix shape: ", A.shape[0])
        print('Diag Vaues: ', A.diagonal(i))
        meanValA = np.mean(A.diagonal(i))
        meanValB = np.mean(B.diagonal(i))
        print('MeanValA: ', [meanValA, meanValB])
        print('Limit: ', lims)
        ax.plot([meanValA, meanValA],lims, 'g-', alpha=0.25, zorder=0)
        
        if i==0:
            color = 'r-'
            alpha = 1
        else:
            color = 'g-'
            alpha = 0.25
        ax.plot(lims, [meanValB, meanValB], color, alpha=alpha, zorder=0)
        
    # Plot both limits against each other
    ax.plot(lims, lims, 'g-', dashes=[2,2], alpha=0.75, zorder=0)
    ax.set_aspect('equal')
    return ax


def ScatterCovariance(X, Xk, columns):
    """ Plot the entries of Cov(Xk,Xk) vs Cov(X,X) and Cov(X,Xk) vs Cov(X,X)
    :param X: n-by-p matrix of original variables
    :param Xk: n-by-p matrix of knockoffs
    """
    # Create subplots
    fig, axarr = plt.subplots(1, 2, figsize=(14,7))

    # Originality
    XX = np.corrcoef(X.T)
    print("Shape X:", np.shape(X))
    print("Cov XX:", XX)
    XkXk = np.corrcoef(Xk.T)
    print("Shape Xk:", np.shape(Xk))
    print("Cov XkXk:", XkXk)
    allX = np.concatenate((X, Xk), axis=1)
    xxkk = np.corrcoef(allX.T)
    print("Shape allX:", np.shape(allX))
    print("Cov allX:", xxkk)

    exk = X.copy()
    exk[:, 1] = Xk[:, 1]
    print('Shape EXK:', np.shape(exk))
    

    # # Plot data
    fig = plt.figure(figsize=(7, 6), 
           dpi = 150) 
    ax1 = fig.add_subplot(111)
    print('diagnostics:', columns)
    # sns.distplot(X[:, 0], color='red', label='Actual')
    dfx = pd.DataFrame(data=X, columns=columns)
    sns.kdeplot(data=dfx, x=columns[0], y=columns[1], cmap='Blues', alpha=0.80, fill=True, levels=7, color='blue', label='Original')
    # Plot the first bivariate distribution with transparency
    dfxk = pd.DataFrame(data=Xk, columns=columns)
    dfexk = pd.DataFrame(data=exk, columns=columns)
    sns.kdeplot(data=dfexk, x=columns[0], y=columns[1], cmap='Reds', alpha=0.40, fill=True, levels=7, color='red', label='Knockoffs')
    # sns.distplot(Xk[:, 0], color='green', label='Knockoffs')
    ax1.set_xlabel(r'')
    ax1.set_ylabel(r'')
    # Add a custom legend
    legend_elements = [
                        Patch(facecolor=plt.cm.Blues(100), alpha=0.70, edgecolor='k', label=r'$(Z_1, Z_2)$'),
                        Patch(facecolor=plt.cm.Reds(100), alpha=0.85, edgecolor='k', label=r'$(Z_1, \tilde{Z}_2)$')
                        ]
    ax1.legend(handles=legend_elements)
    # ax1.legend()
    ax1.set_title(r'Exchangeability: $(Z_1, Z_2) \overset{d}{=} (Z_1, \tilde{Z}_2)$')
    plt.savefig('distribution.pdf')
    plt.show()
    # plt.clf()

    labels = ['Z$_1$', 'Z$_2$', r'$\tilde{Z}_1$', r'$\tilde{Z}_2$']
    xkdf = pd.DataFrame(data=xxkk, columns=labels)

    plt.figure(figsize=(7, 6), 
           dpi = 150) 
    sns.heatmap(xkdf, cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.tick_params(axis = 'x', labelsize = 15) # x font label size
    plt.tick_params(axis = 'y', labelsize = 15) # y font label size
    plt.show()

    PlotScatterHelper(XX, XkXk, ax=axarr[0])
    # axarr[0].set_xlabel(r'$\hat{G}_{\mathbf{Z}\mathbf{Z}}(i,j)$', fontsize=18,  weight='bold')
    # axarr[0].set_ylabel(r'$\hat{G}_{\tilde{\mathbf{Z}}\tilde{\mathbf{Z}}}(i,j)$', fontsize=18, weight='bold')

    axarr[0].set_xlabel(r'$\Sigma_{\mathbf{Z}\mathbf{Z}}(i,j)$', fontsize=18, weight='bold')
    axarr[0].set_ylabel(r'$\Sigma_{\tilde{\mathbf{Z}}\tilde{\mathbf{Z}}}(i,j)$', fontsize=18, weight='bold')

    # Exchangeability
    p = X.shape[1]
    G = np.corrcoef(X.T, Xk.T)
    print('G: ', G)
    XX  = G[:p,:p]
    print("Exchangeability")
    print("Cov XX:", XX)
    XXk = G[:p,p:(2*p)]
    print("Cov XXk:", XXk)

    PlotScatterHelper(XX, XXk, ax=axarr[1])
    # axarr[1].set_xlabel(r'$\hat{G}_{\mathbf{Z}\mathbf{Z}}(i,j)$', fontsize=18, weight='bold')
    # axarr[1].set_ylabel(r'$\hat{G}_{\mathbf{Z}\tilde{\mathbf{Z}}}(i,j)$', fontsize=18, weight='bold')

    axarr[1].set_xlabel(r'$\Sigma_{\mathbf{Z}\mathbf{Z}}(i,j)$', fontsize=18, weight='bold')
    axarr[1].set_ylabel(r'$\Sigma_{\mathbf{Z}\tilde{\mathbf{Z}}}(i,j)$', fontsize=18, weight='bold')
    plt.savefig('cov.png')
    plt.show()
    # return fig


def COV(Z1,Z2,alphas=None):

    """
    Unbiased estimate of the squared Frobenius
    norm difference between the covariance
    matrices of Z1 and Z2
    :param Z1: n-by-p matrix
    :param Z2: n-by-p matrix
    """
    assert(Z1.shape[1]==Z2.shape[1])
    n1 = Z1.shape[0]
    n2 = Z2.shape[0]
    # Center the data
    Z1 = Z1 - Z1.mean(0)
    Z2 = Z2 - Z2.mean(0)
    # Estimate the trace of Sigma1^2
    ZZ1 = torch.mm(Z1, Z1.t())
    A1 = (ZZ1-torch.diag(torch.diag(ZZ1))).pow(2).mean() * n1 / (n1-1)
    # Estimate the trace of Sigma2^2
    ZZ2 = torch.mm(Z2, Z2.t())
    A2 = (ZZ2-torch.diag(torch.diag(ZZ2))).pow(2).mean() * n2 / (n2-1)
    # Estimate  the trace of Sigma1 * Sigma2
    C = torch.mm(Z1,Z2.t()).pow(2).mean()
    # Compute statistic
    T = A1 + A2 - 2.0 * C
    return T.cpu().item()


def MMD(Z1, Z2, alphas):

    """
    Unbiased estimate of the maximum mean discrepancy
    between the distributions of Z1 and Z2
    :param Z1: n-by-p matrix
    :param Z2: n-by-p matrix
    :param alphas: vector of kernel widths
    """
    return mix_rbf_mmd2(Z1, Z2, alphas, biased=False).item()


def KNN(Z1, Z2, alphas, K=1):

    """
    K-nearest neighbor statistic for equality in distribution between Z1 and Z2
    :param Z1: n-by-p matrix
    :param Z2: n-by-p matrix
    """
    n = Z1.shape[0]
    test = nondiff_tests.KNNStatistic(n, n, k=K)
    test_value = test(Z1, Z2).item() / (K*2.0*n)
    return test_value


def Energy(Z1, Z2, alphas=None):

    """
    Energy statistic for equality in distribution between Z1 and Z2
    :param Z1: n-by-p matrix
    :param Z2: n-by-p matrix
    """
    n = Z1.shape[0]
    test = diff_tests.EnergyStatistic(n, n)
    return test(Z1, Z2, ret_matrix=False).item()


def compute_diagnostics(X, Xk, alphas, verbose=True):

    """
    Compute multiple knockoff diagnosics
    :param X: n-by-p matrix of original variables
    :param Xk: n-by-p matrix of knockoffs
    :param alphas: vector of kernel widths for the MMD estimate
    """
    # Divide data into batches
    n1 = int(X.shape[0]/2)
    X1,Xk1 = X[:n1], Xk[:n1]
    X2,Xk2 = X[n1:], Xk[n1:]

    tests = {'Covariance':COV, 'MMD':MMD, 'KNN':KNN, 'Energy':Energy}
    p = X1.shape[1]

    # Initialize results
    results = pd.DataFrame(columns=['Metric', 'Swap', 'Value'])

    # Compute self-correlations
    Ghat = np.corrcoef(X,Xk,rowvar=False)
    selfcorr = np.mean(np.abs(np.diag(Ghat[0:p,p:(2*p)])))
    if verbose:
        print("(Self-corr) : %.6f" %selfcorr)
    results = results.append({'Metric':'Covariance', 'Swap':'self', 'Value':selfcorr}, ignore_index=True)

    # Concatenate observations
    Z_ref = torch.cat((X1,Xk1),1)
    Z_ref2 = torch.cat((X2,Xk2),1)

    # Compute two-sample tests with full swap
    Z_fsw = torch.cat((Xk2,X2),1)
    for test_name,test in tests.items():
        value = test(Z_ref, Z_fsw, alphas)
        if verbose:
            print("%8s. %12s: %.6f" %(test_name, "Full swap", value))
        results = results.append({'Metric':test_name, 'Swap':'full', 'Value':value}, ignore_index=True)

    # Compute two-sample tests with partial swap
    for test_name,test in tests.items():
        swap_inds = np.where(np.random.binomial(1,0.5,size=p))[0]
        Z_psw = Z_ref2.clone()
        Z_psw[:,swap_inds] = Xk2[:,swap_inds]
        Z_psw[:,swap_inds+p] = X2[:,swap_inds]
        value = test(Z_ref, Z_psw, alphas)

        if verbose:
            print("%8s. %12s: %.6f" %(test_name, "Partial swap", value))
        results = results.append({'Metric':test_name, 'Swap':'partial', 'Value':value}, ignore_index=True)

    return results


class KnockoffExam:
    def __init__(self, modelX, machines):

        """
        Initialize knockoff goodness-of-fit tests on independent data
        :param modelX: data generator
        :param machines: named list of knockoff generators
        """

        self.modelX = modelX
        self.machines = machines
        self.use_cuda = torch.cuda.is_available()

        # Parameters for diagnostics
        self.alphas = [1.,2.,4.,8.,16.,32.,64.,128.]

    def diagnose(self, n, n_exams, verbose=False):

        """
        Generate data and knockoffs, then compute goodness-of-fit diagnostics
        :param n: sample size for diagnostics
        :param n_exams: number of samples to compute diagnostics on
        :param verbose: whether to print show values of test statistics in real time
        """

        Results = pd.DataFrame(columns=['Method', 'Metric', 'Swap', 'Value', 'Sample'])
        print("Computing knockoff diagnostics...")
        for exam in range(n_exams):
            sys.stdout.write('\r')
            sys.stdout.write("[%-25s] %d%%" % ('='*int((exam+1)/n_exams*25), ((exam+1)/n_exams)*100))
            sys.stdout.flush()

            # Sample two batches of data simultaneously
            X_np = self.modelX.sample(2*n, test=True)

            for machine_name,machine in self.machines.items():
                if(machine is None):
                    continue

                # Sample knockoff copies
                if(machine == "joint-sampling"):
                    Xk = torch.from_numpy(Xk_oracle).float()
                elif(machine == "independent"):
                    Xk = torch.from_numpy(self.modelX.sample(X.shape[0])).float()
                else:
                    Xk = torch.from_numpy(machine.generate(X_np)).float()

                # Move data and knockoffs onto cuda, if available
                X = torch.from_numpy(X_np).float()
                if(self.use_cuda):
                    X = X.cuda()
                    Xk = Xk.cuda()

                # Compute diagnostics of knockoff quality
                new_res = compute_diagnostics(X, Xk, self.alphas, verbose=verbose)
                new_res["Method"] = machine_name
                new_res["Sample"] = exam

                # Save results
                Results = Results.append(new_res)

        return Results
