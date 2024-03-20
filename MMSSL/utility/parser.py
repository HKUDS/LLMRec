import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="")

    #use less
    parser.add_argument('--verbose', type=int, default=5, help='Interval of evaluation.')    
    parser.add_argument('--core', type=int, default=5, help='5-core for warm-start; 0-core for cold start')
    parser.add_argument('--lambda_coeff', type=float, default=0.9, help='Lambda value of skip connection')

    parser.add_argument('--early_stopping_patience', type=int, default=7, help='') 
    parser.add_argument('--layers', type=int, default=1, help='Number of feature graph conv layers')  
    parser.add_argument('--mess_dropout', nargs='?', default='[0.1, 0.1]', help='Keep probability w.r.t. message dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')
    parser.add_argument('--sparse', type=int, default=1, help='Sparse or dense adjacency matrix')   

    parser.add_argument('--test_flag', nargs='?', default='part', help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')
    parser.add_argument('--metapath_threshold', default=2, type=int, help='metapath_threshold') 
    parser.add_argument('--sc', type=float, default=1.0, help='GCN self connection')
    parser.add_argument('--ssl_c_rate', type=float, default=1.3, help='ssl_c_rate')
    parser.add_argument('--ssl_s_rate', type=float, default=0.8, help='ssl_s_rate')
    parser.add_argument('--g_rate', type=float, default=0.000029, help='ssl_s_rate')
    parser.add_argument('--sample_num', default=1, type=int, help='sample_num') 
    parser.add_argument('--sample_num_neg', default=1, type=int, help='sample_num') 
    parser.add_argument('--sample_num_ii', default=8, type=int, help='sample_num') 
    parser.add_argument('--sample_num_co', default=2, type=int, help='sample_num') 
    parser.add_argument('--mask_rate', default=0.75, type=float, help='sample_num') 
    parser.add_argument('--gss_rate', default=0.85, type=float, help='gene_self_subgraph_rate') 
    parser.add_argument('--anchor_rate', default=0.75, type=float, help='anchor_rate') 
    parser.add_argument('--feat_reg_decay', default=1e-5, type=float, help='feat_reg_decay') 
    parser.add_argument('--ad1_rate', default=0.2, type=float, help='ad1_rate') 
    parser.add_argument('--ad2_rate', default=0.2, type=float, help='ad1_rate')     
    parser.add_argument('--ad_sampNum', type=int, default=1, help='ad topk')  
    parser.add_argument('--ad_topk_multi_num', type=int, default=100, help='ad topk')  
    parser.add_argument('--fake_gene_rate', default=0.0001, type=float, help='fake_gene_rate') 
    parser.add_argument('--ID_layers', type=int, default=1, help='Number of item graph conv layers')  
    parser.add_argument('--reward_rate', default=1, type=float, help='fake_gene_rate') 
    parser.add_argument('--G_embed_size', type=int, default=64, help='Embedding size.')   
    parser.add_argument('--model_num', default=2, type=float, help='fake_gene_rate') 
    parser.add_argument('--negrate', default=0.01, type=float, help='item_neg_sample_rate')
    parser.add_argument('--cis', default=25, type=int, help='') 
    parser.add_argument('--confidence', default=0.5, type=float, help='') 
    parser.add_argument('--ii_it', default=15, type=int, help='') 
    parser.add_argument('--isload', default=False , type=bool, help='whether load model')  #
    parser.add_argument('--isJustTest', default=False , type=bool, help='whether load model')
    parser.add_argument('--loadModelPath', default='/home/ww/Code/work3/BSTRec/Model/retailrocket/for_meta_hidden_dim_dim__8_retailrocket_2021_07_10__18_35_32_lr_0.0003_reg_0.01_batch_size_1024_gnn_layer_[16,16,16].pth', type=str, help='loadModelPath')
    parser.add_argument('--title', default="try_to_draw_line", type=str, help='')  #


    #train
    parser.add_argument('--data_path', nargs='?', default='', help='Input data path.')
    parser.add_argument('--seed', type=int, default=2022, help='Random seed')
    parser.add_argument('--dataset', nargs='?', default='baby', help='Choose a dataset from {sports, baby, clothing, tiktok, allrecipes}')
    parser.add_argument('--epoch', type=int, default=1000, help='Number of epoch.')  #default: 1000
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size.')
    parser.add_argument('--embed_size', type=int, default=64,help='Embedding size.')                     
    parser.add_argument('--D_lr', type=float, default=3e-4, help='Learning rate.')
    parser.add_argument('--topk', type=int, default=10, help='K value of k-NN sparsification')  
    parser.add_argument('--cf_model', nargs='?', default='slmrec', help='Downstream Collaborative Filtering model {mf, ngcf, lightgcn, vbpr, hafr, clcrec, slmrec}')   
    parser.add_argument('--debug', action='store_true')  
    parser.add_argument('--cl_rate', type=float, default=0.03, help='Control the effect of the contrastive auxiliary task')        
    parser.add_argument('--norm_type', nargs='?', default='sym', help='Adjacency matrix normalization operation') 
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU id')
    parser.add_argument('--Ks', nargs='?', default='[10, 20, 50]', help='K value of ndcg/recall @ k')
    parser.add_argument('--regs', nargs='?', default='[1e-5,1e-5,1e-2]', help='for emb_loss.')  #default: '[1e-5,1e-5,1e-2]'
    parser.add_argument('--lr', type=float, default=0.00055, help='Learning rate.')
    parser.add_argument('--emm', default=1e-3, type=float, help='for feature embedding bpr')  #
    parser.add_argument('--L2_alpha', default=1e-3, type=float, help='')  #
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='for opt_D')  #


    #GNN
    parser.add_argument('--drop_rate', type=float, default=0.2, help='dropout rate')
    parser.add_argument('--model_cat_rate', type=float, default=0.55, help='model_cat_rate')
    parser.add_argument('--gnn_cat_rate', type=float, default=0.55, help='gnn_cat_rate')
    parser.add_argument('--id_cat_rate', type=float, default=0.36, help='before GNNs')
    parser.add_argument('--id_cat_rate1', type=float, default=0.36, help='id_cat_rate')
    parser.add_argument('--head_num', default=4, type=int, help='head_num_of_multihead_attention. For multi-model relation.')  #
    parser.add_argument('--dgl_nei_num', default=8, type=int, help='dgl_nei_num')  #


    #GAN
    parser.add_argument('--weight_size', nargs='?', default='[64, 64]', help='Output sizes of every layer')  #default: '[64, 64]'
    parser.add_argument('--G_rate', default=0.0001, type=float, help='for D model1')  #
    parser.add_argument('--G_drop1', default=0.31, type=float, help='for D model2')  #
    parser.add_argument('--G_drop2', default=0.5, type=float, help='')  #
    parser.add_argument('--gp_rate', default=1, type=float, help='gradient penal')  #

    parser.add_argument('--real_data_tau', default=0.005, type=float, help='for real_data soft')  #
    parser.add_argument('--ui_pre_scale', default=100, type=int, help='ui_pre_scale')  


    #cl
    parser.add_argument('--T', default=1, type=int, help='it for ui update')  
    parser.add_argument('--tau', default=0.5, type=float, help='')  #
    parser.add_argument('--geneGraph_rate', default=0.1, type=float, help='')  #
    parser.add_argument('--geneGraph_rate_pos', default=2, type=float, help='')  #
    parser.add_argument('--geneGraph_rate_neg', default=-1, type=float, help='')  #     
    parser.add_argument('--m_topk_rate', default=0.0001, type=float, help='for reconstruct')  
    parser.add_argument('--log_log_scale', default=0.00001, type=int, help='log_log_scale')  
    parser.add_argument('--point', default='', type=str, help='point')  

    return parser.parse_args()


