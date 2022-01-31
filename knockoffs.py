import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from DeepKnockoffs import KnockoffMachine
from DeepKnockoffs import GaussianKnockoffs
import data
import pandas as pd
import diagnostics
import parameters


class Knockoffs:

    def __init__(self):
        self.n = 123

    def GenKnockoffs(self, n, dim, datax):

        import data
        import diagnostics

        # Number of features
        p = dim

        # Load the built-in Gaussian model and its default parameters
        # The currently available built-in models are:
        # - gaussian : Multivariate Gaussian distribution
        # - gmm      : Gaussian mixture model
        # - mstudent : Multivariate Student's-t distribution
        # - sparse   : Multivariate sparse Gaussian distribution
        model = "gaussian"
        distribution_params = parameters.GetDistributionParams(model, p)
        # Initialize the data generator
        DataSampler = data.DataSampler(distribution_params)
        DataSampler = data.DataSampler(distribution_params)

        # Number of training examples
        n = 2000

        # Sample training data
        # X_train = DataSampler.sample(n)
        X_train = datax[0:round(len(datax)*1.0), :]
        # print("Train shape:", X_train.shape)

        # print("Generated a training dataset of size: {} x {}.".format(X_train.shape, X_train.shape))

        # Compute the empirical covariance matrix of the training data
        SigmaHat = np.cov(X_train, rowvar=False)

        # Initialize generator of second-order knockoffs
        second_order = GaussianKnockoffs(SigmaHat, mu=np.mean(X_train, 0), method="sdp")

        # Measure pairwise second-order knockoff correlations
        corr_g = (np.diag(SigmaHat) - np.diag(second_order.Ds)) / np.diag(SigmaHat)

        # print('Average absolute pairwise correlation: %.3f.' % (np.mean(np.abs(corr_g))))

        # Load the default hyperparameters for this model
        training_params = parameters.GetTrainingHyperParams(model)

        # Set the parameters for training deep knockoffs
        pars = dict()
        # Number of epochs
        pars['epochs'] = 5
        # Number of iterations over the full data per epoch
        pars['epoch_length'] = 100
        # Data type, either "continuous" or "binary"
        pars['family'] = "continuous"
        # Dimensions of the data
        pars['p'] = p
        # Size of the test set
        pars['test_size'] = int(0.1 * n)
        # Batch size
        pars['batch_size'] = int(0.45 * n)
        # Learning rate
        pars['lr'] = 0.01
        # When to decrease learning rate (unused when equal to number of epochs)
        pars['lr_milestones'] = [pars['epochs']]
        # Width of the network (number of layers is fixed to 6)
        pars['dim_h'] = int(10 * p)
        # Penalty for the MMD distance
        pars['GAMMA'] = training_params['GAMMA']
        # Penalty encouraging second-order knockoffs
        pars['LAMBDA'] = training_params['LAMBDA']
        # Decorrelation penalty hyperparameter
        pars['DELTA'] = training_params['DELTA']
        # Target pairwise correlations between variables and knockoffs
        pars['target_corr'] = corr_g
        # Kernel widths for the MMD measure (uniform weights)
        pars['alphas'] = [1., 2., 4., 8., 16., 32., 64., 128.]

        # Initialize the machine
        # machine = KnockoffMachine(pars)

        # Train the machine
        # print("Fitting the knockoff machine...")
        # machine.train(X_train)

        # Generate deep knockoffs
        # Xk_train_m = machine.generate(X_train)
        # print("Size of the deep knockoff dataset: %d x %d." % (Xk_train_m.shape))

        # Generate second-order knockoffs
        Xk_train_g = second_order.generate(X_train)
        # print("Size of the second-order knockoff dataset: %d x %d." % (Xk_train_g.shape))

        # # Plot diagnostics for deep knockoffs
        # diagnostics.ScatterCovariance(X_train, Xk_train_m)

        # Sample test data
        # X_test = DataSampler.sample(n, test=True)
        X_test = datax[: round(len(datax)*1.0), :]
        # print("Test shape:", X_test.shape)
        # print("Generated a test dataset of size: %d x %d." % (X_test.shape))

        # Generate deep knockoffs
        # Xk_test_m = machine.generate(X_test)
        # print("Size of the deep knockoff test dataset: %d x %d." % (Xk_test_m.shape))
        # print("Deep Knockoffs: \n", Xk_test_m)

        # Generate second-order knockoffs
        Xk_test_g = second_order.generate(X_test)
        # print("Size of the second-order knockoff test dataset: %d x %d." % (Xk_test_g.shape))

        # Generate oracle knockoffs
        # oracle = GaussianKnockoffs(DataSampler.Sigma, method="sdp", mu=DataSampler.mu)
        # Xk_test_o = oracle.generate(X_test)
        # print("Size of the oracle knockoff test dataset: %d x %d." % (Xk_test_o.shape))
        #
        # Plot diagnostics for deep knockoffs
        # diagnostics.ScatterCovariance(X_test, Xk_test_m)

        # Plot diagnostics for second-order knockoffs
        # diagnostics.ScatterCovariance(X_test, Xk_test_g)

        # Plot diagnostics for oracle knockoffs
        # diagnostics.ScatterCovariance(X_test, Xk_test_o)

        # Compute goodness of fit diagnostics on 50 test sets containing 100 observations each
        n_exams = 50
        n_samples = 100
        # exam = diagnostics.KnockoffExam(DataSampler,
        #                                 {'Second-order': second_order})    # 'Machine': machine, 'Second-order': second_order
        # diagnostics = exam.diagnose(n_samples, n_exams)

        # Summarize diagnostics
        # diagnostics.groupby(['Method', 'Metric', 'Swap']).describe()
        # print(diagnostics.head())

        # Plot covariance goodness-of-fit statistics
        # data = diagnostics[(diagnostics.Metric == "Covariance") & (diagnostics.Swap != "self")]
        # fig, ax = plt.subplots(figsize=(12, 6))
        # sns.boxplot(x="Swap", y="Value", hue="Method", data=data)
        # plt.title("Covariance goodness-of-fit")
        # plt.show()

        # # Plot k-nearest neighbors goodness-of-fit statistics
        # data = diagnostics[(diagnostics.Metric == "KNN") & (diagnostics.Swap != "self")]
        # fig, ax = plt.subplots(figsize=(12, 6))
        # sns.boxplot(x="Swap", y="Value", hue="Method", data=data)
        # plt.title("K-Nearest Neighbors goodness-of-fit")
        # plt.show()
        return Xk_test_g


if __name__ == '__main__':

    def normalize(var):
        nvar = (np.array(var) - np.mean(var)) / np.std(var)
        return nvar


    def deseasonalize(var, interval):

        deseasonalize_data = []
        for i in range(interval, len(var)):
            value = var[i] - var[i - interval]
            deseasonalize_data.append(value)
        return deseasonalize_data


    def down_sample(data, win_size):
        agg_data = []
        monthly_data = []
        for i in range(len(data)):
            monthly_data.append(data[i])
            if (i % win_size) == 0:
                agg_data.append(sum(monthly_data) / win_size)
                monthly_data = []
        return agg_data


    win_size = 1
    # Load synthetic data
    syndata = pd.read_csv("/home/ahmad/PycharmProjects/deepCause/datasets/ncdata/artificial_data_seasonal.csv")
    rg = normalize(down_sample(syndata['Rg'], win_size))
    temp = normalize(down_sample(syndata['T'], win_size))
    gpp = normalize(down_sample(syndata['GPP'], win_size))
    reco = normalize(down_sample(syndata['Reco'], win_size))


    obj = knockoffs()
    datax = np.array([rg, temp, gpp, reco]).transpose()
    n = len(rg)
    dim = 4
    knockoffs = obj.GenKnockoffs(n, dim, datax)
    knockoffs = np.array(knockoffs)
    # print("Deep Knockoffs: \n", knockoffs)

    plt.plot(np.arange(0, 987), rg[0:987], knockoffs[:, 0])
    plt.show()