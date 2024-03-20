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

class G_Model(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim, weight_size, dropout_list, image_feats, text_feats):

        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.weight_size = weight_size
        self.n_ui_layers = len(self.weight_size)
        self.weight_size = [self.embedding_dim] + self.weight_size
        self.image_feats = image_feats
        self.text_feats = text_feats

        initializer = nn.init.xavier_uniform_
        self.act = nn.ReLU()  #nn.PReLU(), nn.LeakyReLU(negative_slope=args.slope), nn.Sigmoid()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=args.drop_rate)
        self.batch_norm = nn.BatchNorm1d(self.embedding_dim)
        self.softmax = nn.Softmax(dim=1)

        self.image_embedding = nn.Embedding.from_pretrained(torch.Tensor(image_feats), freeze=False)
        self.text_embedding = nn.Embedding.from_pretrained(torch.Tensor(text_feats), freeze=False)

        self.image_common_transformation = nn.Linear(image_feats.shape[1], args.embed_size)  #
        self.text_common_transformation = nn.Linear(text_feats.shape[1], args.embed_size)  #
        init.xavier_uniform_(self.image_common_transformation.weight, gain=1.414)  #
        init.xavier_uniform_(self.text_common_transformation.weight, gain=1.414)  #

        self.user_common_feature_embedding = {"image":None,"text":None}
        self.item_common_feature_embedding = {"image":None,"text":None}
        self.user_common_feature_embedding_f = {"image":None,"text":None}
        self.item_common_feature_embedding_f = {"image":None,"text":None}

        self.ssl_common = nn.Bilinear(self.embedding_dim, self.embedding_dim, 1)
        self.ssl_image = nn.Bilinear(self.embedding_dim, self.embedding_dim, 1)
        self.ssl_text = nn.Bilinear(self.embedding_dim, self.embedding_dim, 1)
        init.xavier_uniform_(self.ssl_common.weight, gain=1.414)
        init.xavier_uniform_(self.ssl_image.weight, gain=1.414)
        init.xavier_uniform_(self.ssl_text.weight, gain=1.414)

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


    def para_dict_to_tenser(self, para_dict):  
        """
        :param para_dict: nn.ParameterDict()
        :return: tensor
        """
        tensors = []

        for k in para_dict.keys():
            tensors.append(para_dict[k])

        tensors = torch.stack(tensors, dim=0)

        return tensors


    def behavior_attention(self, embedding_input):  
        embedding = self.para_dict_to_tenser(embedding_input)  
        attention = torch.matmul(embedding, self.weight_dict['w_d_d'])  
        attention = F.softmax(attention, dim=0)*2.5 

        Z = torch.mul(attention, embedding)  
        Z = torch.sum(Z, dim=0)  #
        Z = self.act(Z)

        return Z

    def forward(self, image_adj_norm, image_adj, text_adj_norm, text_adj, ui_graph, iu_graph):

        image_embedding = self.image_embedding.weight
        text_embedding = self.text_embedding.weight
        self.item_common_feature_embedding["image"] = self.image_common_transformation(image_embedding)  
        self.item_common_feature_embedding["text"]  =  self.text_common_transformation(text_embedding)  

        idx_image = np.random.permutation(self.n_items)
        idx_text = np.random.permutation(self.n_items)
        shuffle_image = self.image_feats[idx_image, :]
        shuffle_text = self.text_feats[idx_text, :]
        false_item_feature_embedding_image =  self.image_common_transformation(torch.tensor(shuffle_image).float().cuda())
        false_item_feature_embedding_text = self.text_common_transformation(torch.tensor(shuffle_text).float().cuda())


        for i in range(args.layers):
            #ii
            self.item_common_feature_embedding["image"] = torch.mm(image_adj, self.item_common_feature_embedding['image'])
            self.item_common_feature_embedding["text"] = torch.mm(text_adj, self.item_common_feature_embedding['text'])

            self.item_common_feature_embedding_f["image"] = torch.mm(image_adj, false_item_feature_embedding_image)
            self.item_common_feature_embedding_f["text"] = torch.mm(text_adj, false_item_feature_embedding_text)

            #ui
            self.user_common_feature_embedding["image"] = torch.mm(ui_graph, self.item_common_feature_embedding["image"])
            self.user_common_feature_embedding["text"] = torch.mm(ui_graph, self.item_common_feature_embedding["text"])

            self.user_common_feature_embedding_f["image"] = torch.mm(ui_graph, self.item_common_feature_embedding_f["image"])
            self.user_common_feature_embedding_f["text"] = torch.mm(ui_graph, self.item_common_feature_embedding_f["text"])


        item_common_feature_embedding = (self.item_common_feature_embedding["image"] + self.item_common_feature_embedding["text"]) / 2


        global_item_common_feature_embedding_image = torch.sum(self.item_common_feature_embedding["image"], dim=0)  #
        global_item_common_feature_embedding_text = torch.sum(self.item_common_feature_embedding["text"], dim=0)  #

        global_item_image_specific_feature_embedding = self.sigmoid(global_item_common_feature_embedding_image)
        global_item_text_specific_feature_embedding = self.sigmoid(global_item_common_feature_embedding_text)

        global_item_common_feature_embedding = torch.sum(item_common_feature_embedding, dim=0) 
        global_item_common_feature_embedding = self.sigmoid(global_item_common_feature_embedding)

        global_item_common_feature_embedding = torch.unsqueeze(global_item_common_feature_embedding, 0)
        global_item_common_feature_embedding = global_item_common_feature_embedding.repeat(self.n_items*2, 1)  #

        local_item_common_feature_embedding_t = torch.cat((self.item_common_feature_embedding["image"], self.item_common_feature_embedding["text"]), 0)
        local_item_common_feature_embedding_f = torch.cat((false_item_feature_embedding_image, false_item_feature_embedding_text), 0)

        ssl_common_image_t = torch.unsqueeze(torch.squeeze(self.ssl_common(global_item_common_feature_embedding, local_item_common_feature_embedding_t), 1), 0)
        ssl_common_image_f = torch.unsqueeze(torch.squeeze(self.ssl_common(global_item_common_feature_embedding, local_item_common_feature_embedding_f), 1), 0)

        ssl_common_logit = torch.cat((ssl_common_image_t, ssl_common_image_f),1)

        global_item_image_specific_feature_embedding = self.sigmoid(global_item_common_feature_embedding_image)
        global_item_text_specific_feature_embedding = self.sigmoid(global_item_common_feature_embedding_text)

        global_item_image_specific_feature_embedding = torch.unsqueeze(global_item_image_specific_feature_embedding, 0)
        global_item_image_specific_feature_embedding = global_item_image_specific_feature_embedding.repeat(self.n_items, 1)  
        global_item_text_specific_feature_embedding = torch.unsqueeze(global_item_text_specific_feature_embedding, 0)
        global_item_text_specific_feature_embedding = global_item_text_specific_feature_embedding.repeat(self.n_items, 1)  


        ssl_image_t = torch.unsqueeze(torch.squeeze(self.ssl_image(global_item_image_specific_feature_embedding, self.item_common_feature_embedding["image"]), 1), 0)
        ssl_image_f = torch.unsqueeze(torch.squeeze(self.ssl_image(global_item_image_specific_feature_embedding, false_item_feature_embedding_image), 1), 0)
        ssl_text_t = torch.unsqueeze(torch.squeeze(self.ssl_text(global_item_text_specific_feature_embedding, self.item_common_feature_embedding["text"]), 1), 0)
        ssl_text_f = torch.unsqueeze(torch.squeeze(self.ssl_text(global_item_text_specific_feature_embedding, false_item_feature_embedding_text), 1), 0)

        ssl_image_logit = torch.cat((ssl_image_t, ssl_image_f),1)
        ssl_text_logit = torch.cat((ssl_text_t, ssl_text_f),1)        
 
        item_final = (self.item_common_feature_embedding["image"] + self.item_common_feature_embedding["text"]) / 2
        user_final = (self.user_common_feature_embedding["image"] + self.user_common_feature_embedding["text"]) / 2

        return item_final, user_final, self.item_common_feature_embedding["image"], self.item_common_feature_embedding["text"], self.user_common_feature_embedding["image"], self.user_common_feature_embedding["text"], ssl_common_logit, ssl_image_logit, ssl_text_logit


class D_Model(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim, weight_size, dropout_list, image_feats, text_feats):

        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.weight_size = weight_size
        self.n_ui_layers = len(self.weight_size)
        self.weight_size = [self.embedding_dim] + self.weight_size

        self.image_decoder = nn.Linear(args.embed_size, image_feats.shape[1])
        self.text_decoder = nn.Linear(args.embed_size, text_feats.shape[1])
        nn.init.xavier_uniform_(self.image_decoder.weight)
        nn.init.xavier_uniform_(self.text_decoder.weight)

        self.decoder = nn.ModuleDict() 
        self.decoder['image_decoder'] = self.image_decoder
        self.decoder['text_decoder'] = self.text_decoder

        self.image_trans = nn.Linear(image_feats.shape[1], args.embed_size)
        self.text_trans = nn.Linear(text_feats.shape[1], args.embed_size)
        nn.init.xavier_uniform_(self.image_trans.weight)
        nn.init.xavier_uniform_(self.text_trans.weight)             
        self.encoder = nn.ModuleDict() 
        self.encoder['image_encoder'] = self.image_trans
        self.encoder['text_encoder'] = self.text_trans

        self.common_trans = nn.Linear(args.embed_size, args.embed_size)
        nn.init.xavier_uniform_(self.common_trans.weight)
        self.align = nn.ModuleDict() 
        self.align['common_trans'] = self.common_trans

        self.user_id_embedding = nn.Embedding(n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(n_items, self.embedding_dim)

        nn.init.xavier_uniform_(self.user_id_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)
        self.image_feats = torch.tensor(image_feats).float().cuda()
        self.text_feats = torch.tensor(text_feats).float().cuda()
        self.image_embedding = nn.Embedding.from_pretrained(torch.Tensor(image_feats), freeze=False)
        self.text_embedding = nn.Embedding.from_pretrained(torch.Tensor(text_feats), freeze=False)

        self.image_gnn_trans = nn.Linear(args.embed_size, args.embed_size)
        self.text_gnn_trans = nn.Linear(args.embed_size, args.embed_size)
        nn.init.xavier_uniform_(self.image_gnn_trans.weight)
        nn.init.xavier_uniform_(self.text_gnn_trans.weight)
   
        self.gnn = nn.ModuleDict() 
        self.gnn['user_id_embedding'] = self.user_id_embedding
        self.gnn['item_id_embedding'] = self.item_id_embedding
        self.gnn['image_embedding'] = self.image_embedding
        self.gnn['text_embedding'] = self.text_embedding
        self.gnn['image_trans'] = self.image_trans
        self.gnn['text_trans'] = self.text_trans
        self.gnn['image_gnn_trans'] = self.image_gnn_trans
        self.gnn['text_gnn_trans'] = self.text_gnn_trans

        self.softmax = nn.Softmax(dim=-1)
        self.act = nn.Sigmoid()  #nn.LeakyReLU(0.1) nn.ReLU() nn.PReLU(), nn.LeakyReLU(negative_slope=args.slope), nn.Sigmoid()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=args.drop_rate)
        self.batch_norm = nn.BatchNorm1d(args.embed_size)
        self.tau = 0.5

        self.other = nn.ModuleDict() 
        self.other['softmax'] = self.softmax
        self.other['act'] = self.act
        self.other['sigmoid'] = self.sigmoid
        self.other['dropout'] = self.dropout
        self.other['batch_norm'] = self.batch_norm

        initializer = nn.init.xavier_uniform_
        self.weight_dict = nn.ParameterDict({
            'w_q': nn.Parameter(initializer(torch.empty([args.embed_size, args.embed_size]))),
            'w_k': nn.Parameter(initializer(torch.empty([args.embed_size, args.embed_size]))),
            'w_v': nn.Parameter(initializer(torch.empty([args.embed_size, args.embed_size]))),
            'w_self_attention_item': nn.Parameter(initializer(torch.empty([args.embed_size, args.embed_size]))),
            'w_self_attention_user': nn.Parameter(initializer(torch.empty([args.embed_size, args.embed_size]))),
            'w_self_attention_cat': nn.Parameter(initializer(torch.empty([args.head_num*args.embed_size, args.embed_size]))),
        })
        self.embedding_dict = {'user':{}, 'item':{}}

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


    def multi_head_self_attention(self, trans_w, embedding_t_1, embedding_t):  
       
        q = self.para_dict_to_tenser(embedding_t)
        v = k = self.para_dict_to_tenser(embedding_t_1)
        beh, N, d_h = q.shape[0], q.shape[1], args.embed_size/args.head_num

        Q = torch.matmul(q, trans_w['w_q'])  
        K = torch.matmul(k, trans_w['w_k'])
        V = v

        Q = Q.reshape(beh, N, args.head_num, int(d_h)).permute(2, 0, 1, 3)  
        K = Q.reshape(beh, N, args.head_num, int(d_h)).permute(2, 0, 1, 3)

        Q = torch.unsqueeze(Q, 2) 
        K = torch.unsqueeze(K, 1)  
        V = torch.unsqueeze(V, 1)  

        att = torch.mul(Q, K) / torch.sqrt(torch.tensor(d_h))  
        att = torch.sum(att, dim=-1) 
        att = torch.unsqueeze(att, dim=-1)  
        att = F.softmax(att, dim=2)  

        Z = torch.mul(att, V)  
        Z = torch.sum(Z, dim=2)  

        Z_list = [value for value in Z]
        Z = torch.cat(Z_list, -1)
        Z = torch.matmul(Z, self.weight_dict['w_self_attention_cat'])

        args.model_cat_rate*F.normalize(Z, p=2, dim=2)
        return Z, att.detach()

    def forward(self, ui_graph, iu_graph, image_ui_graph, image_iu_graph, text_ui_graph, text_iu_graph):

        image_feats = image_item_feats = image_feats_encode = self.dropout(self.image_trans(self.image_feats))
        text_feats = text_item_feats = text_feats_encode = self.dropout(self.text_trans(self.text_feats))

        image_feats_decode = self.image_decoder(image_feats_encode)
        text_feats_decode = self.text_decoder(text_feats_encode)

        idx_image = np.random.permutation(self.n_items)
        idx_text = np.random.permutation(self.n_items)
        shuffle_image = self.image_feats[idx_image, :]
        shuffle_text = self.text_feats[idx_text, :]
        image_feats_f =  self.dropout(self.image_trans(torch.tensor(shuffle_image).float().cuda()))
        text_feats_f = self.dropout(self.text_trans(torch.tensor(shuffle_text).float().cuda()))

        for i in range(args.layers):
            image_user_feats = self.mm(ui_graph, image_feats)
            image_item_feats = self.mm(iu_graph, image_user_feats)
            image_user_id = self.mm(image_ui_graph, self.item_id_embedding.weight)
            image_item_id = self.mm(image_iu_graph, self.user_id_embedding.weight)

            text_user_feats = self.mm(ui_graph, text_feats)
            text_item_feats = self.mm(iu_graph, text_user_feats)

            text_user_id = self.mm(text_ui_graph, self.item_id_embedding.weight)
            text_item_id = self.mm(text_iu_graph, self.user_id_embedding.weight)


        self.embedding_dict['user']['image'] = image_user_id
        self.embedding_dict['user']['text'] = text_user_id
        self.embedding_dict['item']['image'] = image_item_id
        self.embedding_dict['item']['text'] = text_item_id
        user_z, att_u = self.multi_head_self_attention(self.weight_dict, self.embedding_dict['user'], self.embedding_dict['user'])
        item_z, att_i = self.multi_head_self_attention(self.weight_dict, self.embedding_dict['item'], self.embedding_dict['item'])
        user_emb = user_z.mean(0)
        item_emb = item_z.mean(0)
        u_g_embeddings = self.user_id_embedding.weight + args.id_cat_rate*F.normalize(user_emb, p=2, dim=1)
        i_g_embeddings = self.item_id_embedding.weight + args.id_cat_rate*F.normalize(item_emb, p=2, dim=1)

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


        u_g_embeddings = u_g_embeddings + args.model_cat_rate*F.normalize(image_user_feats, p=2, dim=1) + args.model_cat_rate*F.normalize(text_user_feats, p=2, dim=1)
        i_g_embeddings = i_g_embeddings + args.model_cat_rate*F.normalize(image_item_feats, p=2, dim=1) + args.model_cat_rate*F.normalize(text_item_feats, p=2, dim=1)

        return u_g_embeddings, i_g_embeddings, image_item_feats, text_item_feats, image_user_feats, text_user_feats, u_g_embeddings, i_g_embeddings, image_user_id, text_user_id, image_item_id, text_item_id



class Discriminator(nn.Module):
    def __init__(self, dim):
        super(Discriminator, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(dim, int(dim/4)),
            nn.LeakyReLU(True),
            nn.BatchNorm1d(int(dim/4)),
    		nn.Dropout(args.G_drop1),

            nn.Linear(int(dim/4), int(dim/8)),
            nn.LeakyReLU(True),
            nn.BatchNorm1d(int(dim/8)),
    		nn.Dropout(args.G_drop2),

            nn.Linear(int(dim/8), 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        output = 100*self.net(x.float())  
        return output.view(-1)

