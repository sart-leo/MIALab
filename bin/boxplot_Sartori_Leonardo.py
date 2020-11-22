import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


def main():

    dirs = [dI for dI in os.listdir("mia-result/") if os.path.isdir(os.path.join("mia-result/", dI))]
    for d in dirs:
        results = pd.read_csv("mia-result/" + d + "/results.csv", sep=';')

        labels = set(results['LABEL'])

        data = np.zeros((int(len(results)/len(labels)), len(labels)))

        for i, label in enumerate(labels):
            data[:, i] = results.loc[results['LABEL'] == label]['DICE']

        fig, ax = plt.subplots()
        ax.boxplot(data)
        plt.title(d)
        plt.ylabel('DSC')
        ax.set_ylim(0, 1)
        plt.xticks(np.linspace(1, len(labels), len(labels)), labels)
    plt.show()


if __name__ == '__main__':
    main()
