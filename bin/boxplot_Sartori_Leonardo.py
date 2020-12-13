import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import statsmodels.api as sm
from scipy import stats
import pylab




def main():
    #folder = "../mia-results-10-10/"
    folder = "../mia-results-final/"
    data_dice, data_hdr, labels, dirs = loadData(folder)
    #plotDice(data_dice, labels, dirs)
    #plotHdrf(data_hdr, labels, dirs)
    #plotHistogram(data_dice[6,:,:], labels)
    #qqPlot(data_dice, labels)

    ref = dirs.index('Unnormalised')
    testKruskal(data_dice, ref, labels, dirs)


def loadData(folder):
    dirs = [dI for dI in os.listdir(folder) if os.path.isdir(os.path.join(folder, dI))]

    test_el = pd.read_csv(folder + dirs[0] + "/results.csv", sep=';')
    labels_t = set(test_el['LABEL'])
    print(test_el)
    data_dice = np.zeros((len(dirs), int(len(test_el) / len(labels_t)), len(labels_t)))
    data_hdr = np.zeros_like(data_dice)

    for h, d in enumerate(dirs):
        results = pd.read_csv(folder + d + "/results.csv", sep=';')

        labels = set(results['LABEL'])

        for i, label in enumerate(labels):
            data_dice[h, :, i] = results.loc[results['LABEL'] == label]['DICE']
            data_hdr[h, :, i] = results.loc[results['LABEL'] == label]['HDRFDST']

    return data_dice, data_hdr, labels, dirs


def plotDice(data_dice, labels, dirs):
    for i, label in enumerate(labels):
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_axes([0.1, 0.3, 0.8, 0.6])
        plt.title(label)
        ax.boxplot(data_dice[:, :, i].T)
        plt.ylabel('DSC')
        ax.set_xticklabels(dirs, rotation=90)
        ax.set_ylim(0, 1)
    plt.show()


def plotHdrf(data_hdr, labels, dirs):
    for i, label in enumerate(labels):
        fig = plt.figure(figsize=(7,7))
        ax = fig.add_axes([0.1, 0.3, 0.8, 0.6])
        plt.title(label)
        ax.boxplot(data_hdr[:, :, i].T)
        plt.ylabel('HDRFDST')
        ax.set_xticklabels(dirs, rotation=90)
        ax.set_ylim(0, np.max(data_hdr))
    plt.show()


def testKruskal(data, ref, labels, dirs):
    ref_data = data[ref, :, :]
    results_k = np.zeros((len(dirs), len(labels), 2))
    for h, d in enumerate(dirs):
        for i, label in enumerate(labels):
            stat_k, p_val_k = stats.kruskal(ref_data[:, i], data[h, :, i])
            results_k[h, i] = [stat_k, p_val_k]
            if stat_k > 3.84:
                print(d, label)

    print(results_k[results_k[:, :, 0] > 3.84])


def plotHistograms(data, labels, dirs, bins=20):
    for i, label in enumerate(labels):
        for h, d in enumerate(dirs):
            fig = plt.figure(figsize=(7, 7))
            ax = fig.add_axes([0.1, 0.3, 0.8, 0.6])
            plt.title(label)
            ax.hist(data[h, :, i].T, bins=bins)

    plt.show()


def plotHistogram(data, labels, bins=20):
    for i, label in enumerate(labels):
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_axes([0.1, 0.3, 0.8, 0.6])
        plt.title(label)
        plt.xlabel('Dice coefficient')
        plt.ylabel('Frequency')
        ax.hist(data[:, i].T, bins=bins)
    plt.show()



def qqPlot(data, labels):
    for i, label in enumerate(labels):
        sm.qqplot(data[6, :, i], line='q')
        pylab.title(label)
    pylab.show()

if __name__ == '__main__':
    main()
