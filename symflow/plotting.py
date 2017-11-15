import matplotlib.pyplot     as plt
import numpy as np
from scipy.stats import pearsonr

def plot_corr(predicted,
              predicted_test,
              expected,
              expected_test,
              subplot_ind,
              title_,
              label,
              target_error = 0,
              plot_hist = True):

    """ Produces scatterplots plots between predicted and expected target values
        and a histogram of the errors

        Parameters
        ----------
        predicted, predicted_test,expected,expected_test:
            1-d numpy.ndarray; predicted/ expected target values for Training
            and test set
        subplot_ind: list/tuple of 3 ints; subplot indices
        title_: string; subplot title
        label: string; axes label (for example units etc.)
        target_error: float; adds a vertical line in the histogram at x-value
            target_error
        plot_hist: boolean; in addition to scatterplot, plot a histogram
            of the absolute prediction error
    """

    plt.subplot(*subplot_ind)

    plt.title(title_)

    rmse = np.sqrt(np.mean((predicted-expected)**2))
    mae = np.mean(np.abs(predicted-expected))
    max_error = np.max(np.abs(predicted - expected))
    R = pearsonr(predicted, expected)[0][0]

    rmse_t = np.sqrt(np.mean((predicted_test-expected_test)**2))
    mae_t = np.mean(np.abs(predicted_test-expected_test))
    max_error_t = np.max(np.abs(predicted_test - expected_test))
    R_t = pearsonr(predicted_test, expected_test)[0][0]

    plt.plot(expected, predicted, ls ='', marker = '.',
         label = 'Training set \nRMSE = {:.1f}meV \nMAE = {:.1f}mev \nmax. \
error = {:.1f}meV \nR = {:.3f}'.format(rmse*1e3,mae*1e3, max_error*1e3, R))

    plt.plot(expected_test, predicted_test, ls ='', marker = '.',
         label = 'Test set \nRMSE = {:.1f}meV \nMAE = {:.1f}mev \nmax. error = \
{:.1f}meV \nR = {:.3f}'.format(rmse_t*1e3,mae_t*1e3, max_error_t*1e3, R_t))

    plt.xlabel("Expected " + label)
    plt.ylabel("Predicted " + label)
    min_ = np.min([predicted,expected]) - 0.01
    max_ = np.max([predicted,expected]) + 0.01
    plt.plot([min_,max_],[min_,max_], color = 'black', lw = 1)

    plt.xlim(min_,max_)
    plt.ylim(min_,max_)
    plt.legend()

    if plot_hist:
        subplot_ind[-1] += 1
        plt.subplot(*subplot_ind)

        percentage = np.sum(np.abs(predicted-expected) <= target_error)/len(predicted)*100
    #     hist(np.abs(predicted - expected), range =(0, max_error + rmse*.5),
    #          label = "{:3.1f}%: Abs. Error <= {} eV".format(percentage, target_error))
        plt.hist([np.abs(predicted - expected),np.abs(predicted_test - expected_test)],
             histtype='barstacked',
             range =(0, max_error + rmse*.5),
             label = ['Training set', 'Test set'])


        if target_error > 0:
            plt.axvline(target_error,0,100, color = 'black', ls='--')

        plt.xlabel('Abs. error [eV]')
        plt.ylabel('N')
        plt.legend()
