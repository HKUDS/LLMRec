import os
import numpy as np
from time import time
import pickle
import scipy.sparse as sp
from scipy.sparse import csr_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from utility.parser import parse_args
from utility.norm import build_sim, build_knn_normalized_graph
args = parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


class MM_Model(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim, weight_size, dropout_list, image_feats, text_feats, user_init_embedding, item_attribute_dict):

        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.weight_size = weight_size
        self.n_ui_layers = len(self.weight_size)
        self.weight_size = [self.embedding_dim] + self.weight_size
        # self.augmented_II_language_csr = augmented_II_language_csr


        self.image_trans = nn.Linear(image_feats.shape[1], args.embed_size)
        self.text_trans = nn.Linear(text_feats.shape[1], args.embed_size)
        self.user_trans = nn.Linear(user_init_embedding.shape[1], args.embed_size)  #aug
        self.item_trans = nn.Linear(item_attribute_dict['title'].shape[1], args.embed_size)  #aug
        nn.init.xavier_uniform_(self.image_trans.weight)
        nn.init.xavier_uniform_(self.text_trans.weight)   
        nn.init.xavier_uniform_(self.user_trans.weight)
        nn.init.xavier_uniform_(self.item_trans.weight)

        self.user_id_embedding = nn.Embedding(n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_id_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)
        self.image_feats = torch.tensor(image_feats).float().cuda()
        self.text_feats = torch.tensor(text_feats).float().cuda()
        self.user_feats = torch.tensor(user_init_embedding).float().cuda()
        self.item_feats = {}
        for key in item_attribute_dict.keys():                                   #aug
            self.item_feats[key] = torch.tensor(item_attribute_dict[key]).float().cuda() 

        self.softmax = nn.Softmax(dim=-1)
        self.act = nn.Sigmoid()  
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=args.drop_rate)
        self.batch_norm = nn.BatchNorm1d(args.embed_size)
        self.tau = 0.5

        # ### construct i-i relation through threshold #######################################################################
        # self.ii_relation_vector_matrix = {}
        # sparsity_threshold = 0.9  # Example sparsity threshold
        # for key in self.item_feats.keys():
        #     ii_matrix = torch.mm(self.item_feats[key], self.item_feats[key].T)
        #     ii_matrix_sparse = csr_matrix(ii_matrix.cpu().numpy())
        #     ii_matrix_sparse[ii_matrix_sparse < sparsity_threshold] = 0
        #     self.ii_relation_vector_matrix[key] = self.matrix_to_tensor(self.csr_norm(ii_matrix_sparse, mean_flag=False))
        # ### construct i-i relation through threshold #######################################################################

        # ### construct i-i relation through topk #######################################################################
        # topk_num = 2
        # self.ii_relation_vector_matrix = {}
        # for key in self.item_feats.keys():
        #     # Keep only the top-k most similar items for each item
        #     ii_matrix = torch.mm(self.item_feats[key], self.item_feats[key].T)
        #     value, indices = torch.topk(ii_matrix, topk_num)
        #     x_data, y_data = [], []
        #     for i in range(indices.shape[0]):
        #         x_data += np.full((topk_num), i).tolist()
        #         y_data += indices[i].tolist()
        #     data = np.ones(len(x_data))
        #     self.ii_relation_vector_matrix[key] = self.matrix_to_tensor(self.csr_norm(csr_matrix((data,(x_data, y_data)), shape=(1000, 1000)), mean_flag=False))
        # ### construct i-i relation through topk #######################################################################


    def mm(self, x, y):
        if args.sparse:
            return torch.sparse.mm(x, y)
        else:
            return torch.mm(x, y)
    def sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def batched_contrastive_loss(self, z1, z2, batch_size=4096):
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  
            between_sim = f(self.sim(z1[mask], z2))  

            losses.append(-torch.log(
                between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                / (refl_sim.sum(1) + between_sim.sum(1)
                   - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))
                   
        loss_vec = torch.cat(losses)
        return loss_vec.mean()

    def csr_norm(self, csr_mat, mean_flag=False):
        rowsum = np.array(csr_mat.sum(1))
        rowsum = np.power(rowsum+1e-8, -0.5).flatten()
        rowsum[np.isinf(rowsum)] = 0.
        rowsum_diag = sp.diags(rowsum)

        colsum = np.array(csr_mat.sum(0))
        colsum = np.power(colsum+1e-8, -0.5).flatten()
        colsum[np.isinf(colsum)] = 0.
        colsum_diag = sp.diags(colsum)

        if mean_flag == False:
            return rowsum_diag*csr_mat*colsum_diag
        else:
            return rowsum_diag*csr_mat

    def matrix_to_tensor(self, cur_matrix):
        if type(cur_matrix) != sp.coo_matrix:
            cur_matrix = cur_matrix.tocoo()  #
        indices = torch.from_numpy(np.vstack((cur_matrix.row, cur_matrix.col)).astype(np.int64))  #
        values = torch.from_numpy(cur_matrix.data)  #
        shape = torch.Size(cur_matrix.shape)

        return torch.sparse.FloatTensor(indices, values, shape).to(torch.float32).cuda()  #

    def para_dict_to_tenser(self, para_dict):  
        """
        :param para_dict: nn.ParameterDict()
        :return: tensor
        """
        tensors = []

        for beh in para_dict.keys():
            tensors.append(para_dict[beh])
        tensors = torch.stack(tensors, dim=0)

        return tensors


    def forward(self, ui_graph, iu_graph, image_ui_graph, image_iu_graph, text_ui_graph, text_iu_graph):


        # feature mask ##############################################################################
        raw_image_feats = self.image_feats.clone()
        raw_text_feats = self.text_feats.clone()
        raw_item_feats = {}
        for key in self.item_feats.keys():
            raw_item_feats[key] = self.item_feats[key].clone()

        perm = torch.randperm(self.n_items)
        num_mask_nodes = int(args.mask_rate * self.n_items)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes: ]

        self.image_feats[mask_nodes] = self.image_feats.mean(0)
        self.text_feats[mask_nodes] = self.text_feats.mean(0)
        for key in self.item_feats.keys():
            self.item_feats[key][mask_nodes] = self.item_feats[key].mean(0)

        raw_image_feats = self.dropout(self.image_trans(raw_image_feats))
        raw_text_feats = self.dropout(self.text_trans(raw_text_feats))
        for key in self.item_feats.keys():
            raw_item_feats[key] = self.dropout(self.item_trans(raw_item_feats[key]))
        # feature mask ##############################################################################


        image_feats = self.dropout(self.image_trans(self.image_feats))
        text_feats = self.dropout(self.text_trans(self.text_feats))
        user_feats = self.dropout(self.user_trans(self.user_feats.to(torch.float32)))
        item_feats = {}
        for key in self.item_feats.keys():
            item_feats[key] = self.dropout(self.item_trans(self.item_feats[key]))



        for i in range(args.layers):
            image_user_feats = self.mm(ui_graph, image_feats)
            image_item_feats = self.mm(iu_graph, image_user_feats)

            text_user_feats = self.mm(ui_graph, text_feats)
            text_item_feats = self.mm(iu_graph, text_user_feats)

        # aug item attribute
        user_feat_from_item = {}
        for key in self.item_feats.keys():
            user_feat_from_item[key] = self.mm(ui_graph, item_feats[key])
            item_feats[key] = self.mm(iu_graph, user_feat_from_item[key])

        # aug user profile
        item_prof_feat = self.mm(iu_graph, user_feats)
        user_prof_feat = self.mm(ui_graph,  item_prof_feat)

        u_g_embeddings = self.user_id_embedding.weight
        # u_g_embeddings = user_feats
        i_g_embeddings = self.item_id_embedding.weight             

        user_emb_list = [u_g_embeddings]
        item_emb_list = [i_g_embeddings]
        for i in range(self.n_ui_layers):    
            if i == (self.n_ui_layers-1):
                u_g_embeddings = self.softmax( torch.mm(ui_graph, i_g_embeddings) ) 
                i_g_embeddings = self.softmax( torch.mm(iu_graph, u_g_embeddings) )
            else:
                u_g_embeddings = torch.mm(ui_graph, i_g_embeddings) 
                i_g_embeddings = torch.mm(iu_graph, u_g_embeddings) 

            user_emb_list.append(u_g_embeddings)
            item_emb_list.append(i_g_embeddings)

        u_g_embeddings = torch.mean(torch.stack(user_emb_list), dim=0)
        i_g_embeddings = torch.mean(torch.stack(item_emb_list), dim=0) 

        u_g_embeddings = u_g_embeddings + args.model_cat_rate*F.normalize(image_user_feats, p=2, dim=1) + args.model_cat_rate*F.normalize(text_user_feats, p=2, dim=1) #+ 0.1*F.normalize(user_feats, p=2, dim=1)
        i_g_embeddings = i_g_embeddings + args.model_cat_rate*F.normalize(image_item_feats, p=2, dim=1) + args.model_cat_rate*F.normalize(text_item_feats, p=2, dim=1)
        # profile
        u_g_embeddings += args.user_cat_rate*F.normalize(user_prof_feat, p=2, dim=1)
        i_g_embeddings += args.user_cat_rate*F.normalize(item_prof_feat, p=2, dim=1)

        # # II relation
        # i_g_embeddings = torch.mm(self.augmented_II_language_csr, i_g_embeddings)  
        # attribute 
        for key in self.item_feats.keys():
            u_g_embeddings += args.item_cat_rate*F.normalize(user_feat_from_item[key], p=2, dim=1)
            i_g_embeddings += args.item_cat_rate*F.normalize(item_feats[key], p=2, dim=1) 

        return u_g_embeddings, i_g_embeddings, image_item_feats, text_item_feats, image_user_feats, text_user_feats, user_feats, item_feats, user_prof_feat, item_prof_feat, user_feat_from_item, item_feats, mask_nodes, raw_image_feats, raw_text_feats, raw_item_feats


