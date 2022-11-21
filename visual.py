import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.io import loadmat
import random


def t_sne(embeds, labels, sample_num, epoch,show_fig=True):
    """
    visualize embedding by t-SNE algorithm
    :param embeds: embedding of the data
    :param labels: labels
    :param sample_num: the num of samples
    :param show_fig: if show the figure
    :return: figure
    """
    if sample_num>embeds.shape[0]:
        print("Value Error: Sample larger than population")
        return

    # sampling
    random.seed(1)
    sample_index = random.sample(range(0, embeds.shape[0]), sample_num)
    sample_index.sort()
    sample_embeds = embeds[sample_index]
    sample_labels = labels[sample_index]

    # cluster number
    unique = np.unique(sample_labels)
    clusters = np.size(unique, axis=0)

    # t-SNE
    ts = TSNE(n_components=2, init='pca', random_state=0)
    ts_embeds = ts.fit_transform(sample_embeds[:, :])

    # remove outlier
    mean, std = np.mean(ts_embeds, axis=0), np.std(ts_embeds, axis=0)
    for i in range(len(ts_embeds)):
        if (ts_embeds[i] - mean < 3 * std).all():
            np.delete(ts_embeds, i)

    # normalization
    x_min, x_max = np.min(ts_embeds, 0), np.max(ts_embeds, 0)
    norm_ts_embeds = (ts_embeds - x_min) / (x_max - x_min)

    # plot
    fig = plt.figure()
    for i in range(norm_ts_embeds.shape[0]):
        plt.plot(norm_ts_embeds[i, 0], norm_ts_embeds[i, 1],
                 color=plt.cm.Set1(sample_labels[i] % clusters), marker='.', markersize=7)
    plt.xticks([])
    plt.yticks([])
    #plt.title('t-SNE', fontsize=14)
    plt.axis('off')
    # plt.savefig("C://Users//Admin//Desktop//pygcn-master//OUTPUT//fig//{}.png".format(epoch))
    plt.savefig("C://Users//Admin//Desktop//pygcn-master//OUTPUT//fig//{}.eps".format(epoch),format='eps',dpi=2000)
    return fig



data_path = "../data/synthetic3d.mat"  # 改
data = loadmat(data_path)
X = {}
for i in range(3):
    diff_view = data['X'][i, 0]
    diff_view = np.array(diff_view, dtype=np.float32)
    X.update({str(i): diff_view})
D = X['2']  # 改
#sample_num = 500
sample_num = D.shape[0]
LABELS = np.array(data['Y'])
LABELS = np.squeeze(LABELS)

fig = t_sne(D, LABELS, sample_num,30000)


