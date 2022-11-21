import numpy as np
import scipy.sparse as sp
import torch
import scipy.io as scio
from sklearn import metrics

from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix
import warnings
from pygcn.metric import cal_clustering_metric_f
warnings.filterwarnings('ignore')


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_data(l,u,num_nei):
    """Load the data"""
    path='../data/{}.mat'.format(l)
    data = scio.loadmat(path)
    ##########Y##########
    Y = data['Y']
    Y = np.array(Y)
    Y = np.squeeze(Y)
    ########X和Z#############
    a = data['X']
    X = {}
    v = data['X'].shape[0]
    print("视图个数：", v)
    for i in range(v):
        # print('------------------------第 %5.4f 个视图' % (i + 1), end=' ')
        h = csr_matrix(a[i][0]).toarray()
        c = np.mat(h, dtype=np.float32)
        c = normalize(c)  # 数据矩阵X
        X.update({str(i): c})
        X[str(i)] = torch.FloatTensor(np.array(X[str(i)]))
    n = X[str(0)].shape[0]
    path = '../data/{}.mat'.format(u)
    data2 = scio.loadmat(path)
    zv = data2['ZV']
    ZV = {}  # ZV的列和为1
    for i in range(v):
        c = np.mat(zv[0][i], dtype=np.float32)
        ZV.update({str(i): c})
        ZV[str(i)] = torch.FloatTensor(np.array(ZV[str(i)]))
    A,S_inx = gen_A(num_nei, ZV)
    S_inx = torch.FloatTensor(np.array(S_inx))
    pre = A_TURE(A, Y, num_nei)
    print("matlab的A的精度为：", pre)
    adj=sp.coo_matrix(A,dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    adj= sparse_mx_to_torch_sparse_tensor((adj))

    for i in range(v):
        ZV[str(i)] = ZV[str(i)].T
        for j in range(n):
            ZV[str(i)][j, j] = 0.005
    ZV = gen_z(50, ZV)

    return adj, X,Y,ZV,pre,S_inx
#返回邻接矩阵，特征矩阵，标签，ZV，A的精度

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def A_TURE(A,Y,c):
    """ Test
    the
    accuracy
    of
    A"""

    n=A.shape[0]
    sum_a=0
    for i in range(n):
        for j in range(n):
            if A[i,j]==1 and Y[i]==Y[j]:
                sum_a=sum_a+1
    pre=sum_a/(c*n)
    return  pre



def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def clustering(Y, Z, k_means=True, SC=True):
    """
    Spectral clustering
    """
    n_clusters = np.unique(Y).shape[0]
    #print("n_clusters",n_clusters)
    if k_means:
        embedding = Z.cpu().detach().numpy()
        km = KMeans(n_clusters=n_clusters).fit(embedding)
        prediction = km.predict(embedding)
        acc, nmi,purity, fscore, precision, recall = cal_clustering_metric_f(Y, prediction,n_clusters)
        print('k-means --- ACC: %5.4f, NMI: %5.4f' % (acc, nmi), end=' ')
    if SC:
        degree = torch.sum(Z, dim=1).pow(-0.5)
        L = (Z * degree).t() * degree
        L = L.cpu()
        _, vectors = L.symeig(True)
        indicator = vectors[:, -n_clusters:]
        indicator = indicator / (indicator.norm(dim=1)+10**-10).repeat(n_clusters, 1).t()
        indicator = indicator.cpu().numpy()
        km = KMeans(n_clusters=n_clusters).fit(indicator)
        prediction = km.predict(indicator)
        acc, nmi,purity, fscore, precision, recall = cal_clustering_metric_f(Y, prediction,n_clusters)
        print('SC --- ACC: %5.4f, NMI: %5.4f,purity: %5.4f, fscore: %5.4f, precision: %5.4f, recall: %5.4f' % (acc, nmi, purity, fscore, precision, recall))
        return acc, nmi,purity, fscore, precision, recall


#####################生成A和邻居化的z###############################

def gen_z(k,Z):
    """
    Generate a structured affinity matrix
    """
    Z_nei = {}
    index_k = {}
    for v in range(len(Z)):
        Z[str(v)] = np.array(Z[str(v)])
        # Lst = List[:]  # 对列表进行浅复制，避免后面更改原列表数据
        z_k = np.zeros(Z[str(v)].shape)
        for j in range(Z[str(v)].shape[0]):
            # print("j",j)
            Lst2 = Z[str(v)][j].tolist()
            # print(Lst2)
            l = []
            for i in range(k):
                index_i = Lst2.index(max(Lst2))  # 得到列表的最小值，并得到该最小值的索引
                # print("最大值的索引",index_i)
                l.append(index_i)
                Lst2[index_i] = float('-inf')  # 将遍历过的列表最小值改为无穷大，下次不再选择
            z_k [j][l] = Z[str(v)][j][l]
            # index_k.update({str(j): l})
        z_k = normalize(z_k)
        Z_nei.update({str(v): z_k})
        Z_nei[str(v)] = torch.FloatTensor(np.array(Z_nei[str(v)]))
    return Z_nei

def gen_A(m,Z):
    List = np.array(Z[str(0)])
    Z_AVG = np.zeros(List.shape)
    Z_AVG = torch.FloatTensor(np.array( Z_AVG))
    for v in range(len(Z)):
        Z_AVG=Z_AVG+Z[str(v)]
    Z_AVG=Z_AVG/v
    A = []
    S_inx=[]
    A = np.zeros(Z_AVG.shape)
    print(A.shape)
    smp=Z_AVG.shape[0]

    S_inx = np.zeros([smp,m])
    print(S_inx.shape)
    for j in range(Z_AVG.shape[0]):
        Lst2 = Z_AVG[j].tolist()
        l = []
        for i in range(m):
            index_i = Lst2.index(max(Lst2))  # 得到列表的最小值，并得到该最小值的索引
            l.append(index_i)
            Lst2[index_i] = float('-inf')  # 将遍历过的列表最小值改为无穷大，下次不再选择
            if i < m:
                A[j][l] = 1
                S_inx[j][i]=index_i
    return A,S_inx

def Purity_score(y_true, y_pred):
    """Purity score
        Args:
            y_true(np.ndarray): n*1 matrix Ground truth labels
            y_pred(np.ndarray): n*1 matrix Predicted clusters

        Returns:
            float: Purity score
    """
    y_voted_labels = np.zeros(y_true.shape)
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true==labels[k]] = ordered_labels[k]
    labels = np.unique(y_true)
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred==cluster], bins=bins)
        # Find the most present label in the cluster
        winner = np.argmax(hist)
        y_voted_labels[y_pred==cluster] = winner

    purity = metrics.accuracy_score(y_true, y_voted_labels)
    return purity

