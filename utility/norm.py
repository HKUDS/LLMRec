import torch
import numpy as np
from scipy.sparse import csr_matrix 
#defining the build_sim function
def build_sim(context):
    #get the normalized context
    context_norm = context.div(torch.norm(context, p=2, dim=-1, keepdim=True))
    sim = torch.sparse.mm(context_norm, context_norm.transpose(1, 0))
    return sim
#defining the build_knn_graph function
def build_knn_normalized_graph(adj, topk, is_sparse, norm_type):
    #get the device
    device = adj.device
    #get the topk values and indices
    knn_val, knn_ind = torch.topk(adj, topk, dim=-1)  
    n_item = knn_val.shape[0]
    n_data = knn_val.shape[0]*knn_val.shape[1]
    data = np.ones(n_data)
    #if the graph is sparse
    if is_sparse:
        #get the tuple list
        tuple_list = [[row, int(col)] for row in range(len(knn_ind)) for col in knn_ind[row]]  #
        row = [i[0] for i in tuple_list]  #
        col = [i[1] for i in tuple_list]  #
        #create the csr matrix
        ii_graph = csr_matrix((data, (row, col)) ,shape=(n_item, n_item))
        return ii_graph
    else:
        #create the adjacency matrix
        weighted_adjacency_matrix = (torch.zeros_like(adj)).scatter_(-1, knn_ind, knn_val)
        #get the dense laplacian
        return get_dense_laplacian(weighted_adjacency_matrix, normalization=norm_type)

#defining the get_sparse_laplacian function
def get_sparse_laplacian(edge_index, edge_weight, num_nodes, normalization='none'): 
    #get the scatter add function 
    from torch_scatter import scatter_add
    row, col = edge_index[0], edge_index[1]  
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)  
    #get the  edge_index ,edge_weight 
    if normalization == 'sym':
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    elif normalization == 'rw':
        deg_inv = 1.0 / deg
        deg_inv.masked_fill_(deg_inv == float('inf'), 0)
        edge_weight = deg_inv[row] * edge_weight
    return edge_index, edge_weight

#defining the get_dense_laplacian function
def get_dense_laplacian(adj, normalization='none'):
    #get the row sum
    if normalization == 'sym':
        #get the row sum
        rowsum = torch.sum(adj, -1)
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diagflat(d_inv_sqrt)
        #get the normalized laplacian
        L_norm = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
    elif normalization == 'rw':
        #get the row sum
        rowsum = torch.sum(adj, -1)
        d_inv = torch.pow(rowsum, -1)
        d_inv[torch.isinf(d_inv)] = 0.
        #get the diagonal matrix
        d_mat_inv = torch.diagflat(d_inv)
        #get the normalized laplacian
        L_norm = torch.mm(d_mat_inv, adj)
    elif normalization == 'none':
        #get the adjacency matrix
        L_norm = adj
    return L_norm
