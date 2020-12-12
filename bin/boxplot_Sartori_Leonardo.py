import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


def main():
    folder = "../mia-results-final/"
    data_dice, data_hdr, labels, dirs = loadData(folder)
    plotDice(data_dice, labels, dirs)
    plotHdrf(data_hdr, labels, dirs)
    #plotHistograms(data_dice, labels, dirs)


def loadData(folder):
    dirs = [dI for dI in os.listdir(folder) if os.path.isdir(os.path.join(folder, dI))]

    test_el = pd.read_csv(folder + dirs[0] + "/results.csv", sep=';')
    labels_t = set(test_el['LABEL'])
    print(test_el)
    data_dice = np.zeros((len(dirs), int(len(test_el) / len(labels_t)), len(labels_t)))
    data_hdr = np.zeros_like(data_dice)

    for h, d in enumerate(dirs):
        results = pd.read_csv("../mia-results-final/" + d + "/results.csv", sep=';')

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


def plotHistograms(data, labels, dirs):
    for i, label in enumerate(labels):
        for h, d in enumerate(dirs):
            fig = plt.figure(figsize=(7, 7))
            ax = fig.add_axes([0.1, 0.3, 0.8, 0.6])
            plt.title(label)
            ax.hist(data[h, :, i].T, bins=20)
    plt.show()



if __name__ == '__main__':
    main()
