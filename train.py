from __future__ import division
from __future__ import print_function
from torch.utils.tensorboard import SummaryWriter
import warnings
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from pygcn.utils import load_data,clustering
from pygcn.models import GCN


warnings.filterwarnings('ignore')


dataset={'0':'NGs'}
dataset_c={'0':'NGs_c' }

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')#42
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

def train(epoch,features,adj,Q,S_inx,b):####
    outputs = {}
    t = time.time()
    outputs3 = np.zeros((n, n))
    for i in range(v):
        models[str(i)].train()
        optimizers[str(i)].zero_grad()
        output,x_v = models[str(i)](features[str(i)], adj)
        output2 = output.data.cpu().numpy()
        outputs3 = outputs3 + output2
        kl = F.kl_div(output.log(), Q[str(i)], reduction='sum')
        S_re = output
        loss_all_node = torch.FloatTensor([0]).cuda()
        for j in range(n):
            k0 = torch.exp(S_re[j]).sum() - torch.exp(S_re[j][j])
            loss_nbr = torch.FloatTensor([0]).cuda()
            for z in range(5):
                if S_inx[j][z] != j:
                    loss_nbr = loss_nbr - torch.log(torch.exp(S_re[j][int(S_inx[j][z].item())]) / k0)
            loss_all_node = loss_all_node + loss_nbr

        loss_S_re = b * loss_all_node + kl
        print(loss_S_re)
        loss_S_re.backward()
        optimizers[str(i)].step()
    if epoch % 10 ==0:
        output3 = (outputs3 + outputs3.T) / 2
        output3 = torch.FloatTensor(np.array(output3))
        acc_o, nmi_o,purity, fscore, precision, recall = clustering(Y, output3, k_means=False, SC=True)
    print('Epoch: {:04d}'.format(epoch+1),
            'loss_train: {:.4f}'.format(kl.item()),
            'time: {:.4f}s'.format(time.time() - t))

print(len(dataset))
pre_a = []
nei_shu=[5]
b=10
for t in nei_shu:
    print("邻居",t)
    for p in range(0,1):
        print(dataset[str(p)])
        adj, features,Y,ZV,pre,S_inx= load_data(dataset[str(p)],dataset_c[str(p)],t)
        pre_a.append(pre)
        v=len(features)
        n=ZV[str(0)].shape[0]
        models= {}
        optimizers={}
        for i in range(v):
        # Model and optimizer
            model = GCN(nfeat=features[str(i)].shape[1],#输入层神经元个数    维度
                        nhid=args.hidden,#隐藏层神经元个数
                        nclass=features[str(i)].shape[0],#输出层的个数    样本点的个数
                        #nclass=3,
                        dropout=args.dropout)
            models.update({str(i): model})
            optimizer = optim.Adam(model.parameters(),
                                   lr=args.lr, weight_decay=args.weight_decay)
            optimizers.update({str(i): optimizer})
            if args.cuda:
                models[str(i)].cuda()
                features[str(i)] = features[str(i)].cuda()
                adj = adj.cuda()
                ZV[str(i)] = ZV[str(i)].cuda()
                S_inx = S_inx.cuda()

        writer = SummaryWriter(comment='test_your_commment',filename_suffix="test_your_filename_suffix")

            # Train model
        t_total = time.time()
        for epoch in range(args.epochs):
                # acc_r,nmi_r=train(epoch,features[str(i)],adj,labels[str(i)])
            train(epoch, features, adj, ZV,S_inx,b)
        writer.close()
        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))