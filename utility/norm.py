import torch
import numpy as np
from scipy.sparse import csr_matrix 

def build_sim(context):
    context_norm = context.div(torch.norm(context, p=2, dim=-1, keepdim=True))
    sim = torch.sparse.mm(context_norm, context_norm.transpose(1, 0))
    # a, b = context_norm.shape
    # b, c = context_norm.transpose(1, 0).shape
    # ab = context_norm.unsqueeze(-1)  #.repeat(1,1,c)
    # bc = context_norm.transpose(1, 0).unsqueeze(0)  #.repeat(a, 1,1)
    # sim = torch.mul(ab, bc).sum(dim=1, keepdim=False)

    return sim

# def build_knn_normalized_graph(adj, topk, is_sparse, norm_type):
#     device = adj.device
#     knn_val, knn_ind = torch.topk(adj, topk, dim=-1)
#     if is_sparse:
#         tuple_list = [[row, int(col)] for row in range(len(knn_ind)) for col in knn_ind[row]]
#         row = [i[0] for i in tuple_list]
#         col = [i[1] for i in tuple_list]
#         i = torch.LongTensor([row, col]).to(device)
#         v = knn_val.flatten()
#         edge_index, edge_weight = get_sparse_laplacian(i, v, normalization=norm_type, num_nodes=adj.shape[0])
#         return torch.sparse_coo_tensor(edge_index, edge_weight, adj.shape)
#     else:
#         weighted_adjacency_matrix = (torch.zeros_like(adj)).scatter_(-1, knn_ind, knn_val)
#         return get_dense_laplacian(weighted_adjacency_matrix, normalization=norm_type)

def build_knn_normalized_graph(adj, topk, is_sparse, norm_type):
    device = adj.device
    knn_val, knn_ind = torch.topk(adj, topk, dim=-1)  #[7050, 10] [7050, 10]
    n_item = knn_val.shape[0]
    n_data = knn_val.shape[0]*knn_val.shape[1]
    data = np.ones(n_data)
    if is_sparse:
        tuple_list = [[row, int(col)] for row in range(len(knn_ind)) for col in knn_ind[row]]  #[70500]
        # data = np.array(knn_val.flatten().cpu())  #args.topk_rate*
        row = [i[0] for i in tuple_list]  #[70500]
        col = [i[1] for i in tuple_list]  #[70500]
        # #-----------------------------------------------------------------------------------------------------
        # i = torch.LongTensor([row, col]).to(device)
        # v = knn_val.flatten()
        # edge_index, edge_weight = get_sparse_laplacian(i, v, normalization=norm_type, num_nodes=adj.shape[0])
        # #-----------------------------------------------------------------------------------------------------
        ii_graph = csr_matrix((data, (row, col)) ,shape=(n_item, n_item))
        # return torch.sparse_coo_tensor(edge_index, edge_weight, adj.shape)
        return ii_graph
    else:
        weighted_adjacency_matrix = (torch.zeros_like(adj)).scatter_(-1, knn_ind, knn_val)
        return get_dense_laplacian(weighted_adjacency_matrix, normalization=norm_type)


def get_sparse_laplacian(edge_index, edge_weight, num_nodes, normalization='none'):  #[2, 70500], [70500]
    from torch_scatter import scatter_add
    row, col = edge_index[0], edge_index[1]  #[70500] [70500]
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)  #[7050]

    if normalization == 'sym':
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    elif normalization == 'rw':
        deg_inv = 1.0 / deg
        deg_inv.masked_fill_(deg_inv == float('inf'), 0)
        edge_weight = deg_inv[row] * edge_weight
    return edge_index, edge_weight


def get_dense_laplacian(adj, normalization='none'):
    if normalization == 'sym':
        rowsum = torch.sum(adj, -1)
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diagflat(d_inv_sqrt)
        L_norm = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
    elif normalization == 'rw':
        rowsum = torch.sum(adj, -1)
        d_inv = torch.pow(rowsum, -1)
        d_inv[torch.isinf(d_inv)] = 0.
        d_mat_inv = torch.diagflat(d_inv)
        L_norm = torch.mm(d_mat_inv, adj)
    elif normalization == 'none':
        L_norm = adj
    return L_norm
