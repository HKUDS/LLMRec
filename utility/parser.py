import argparse


#---LLMAug netflix---------------------------------------------------------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument('--data_path', nargs='?', default='/home/weiw/Code/LLMs/data/', help='Input data path.')  
    parser.add_argument('--seed', type=int, default=2022, help='Random seed')
    parser.add_argument('--dataset', nargs='?', default='netflix_valid_item', help='Choose a dataset from {preprocessed_raw_MovieLens netflix_valid_item}')
    parser.add_argument('--verbose', type=int, default=5, help='Interval of evaluation.')
    parser.add_argument('--epoch', type=int, default=1000, help='Number of epoch.')  #default: 1000
    parser.add_argument('--regs', nargs='?', default='[1e-5,1e-5,1e-2]', help='Regularizations.')  
    # parser.add_argument('--lr_d', type=float, default=0.0001, help='Learning rate.')  
    parser.add_argument('--embed_size', type=int, default=64, help='Embedding size.')                     
    parser.add_argument('--weight_size', nargs='?', default='[64, 64]', help='Output sizes of every layer')  #default: '[64, 64]'
    parser.add_argument('--early_stopping_patience', type=int, default=7, help='') 
    parser.add_argument('--mess_dropout', nargs='?', default='[0.1, 0.1]',help='Keep probability w.r.t. message dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')
    parser.add_argument('--sparse', type=int, default=1, help='Sparse or dense adjacency matrix')   
    parser.add_argument('--debug', action='store_true')  
    parser.add_argument('--norm_type', nargs='?', default='sym', help='Adjacency matrix normalization operation') 
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU id')
    parser.add_argument('--Ks', nargs='?', default='[10, 20, 50]', help='K value of ndcg/recall @ k')
    parser.add_argument('--test_flag', nargs='?', default='part', help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')
    parser.add_argument('--sc', type=float, default=1.0, help='GCN self connection')
    parser.add_argument('--feat_reg_decay', default=1e-5, type=float, help='feat_reg_decay') 
    parser.add_argument('--isload', default=False , type=bool, help='whether load model')  #布尔值加引号...直接编译为True
    parser.add_argument('--isJustTest', default=False , type=bool, help='whether load model')
    parser.add_argument('--loadModelPath', default='/home/ww/Code/work3/BSTRec/Model/retailrocket/for_meta_hidden_dim_dim__8_retailrocket_2021_07_10__18_35_32_lr_0.0003_reg_0.01_batch_size_1024_gnn_layer_[16,16,16].pth', type=str, help='loadModelPath')
    parser.add_argument('--title', default="try_to_draw_line", type=str, help='')  
    parser.add_argument('--cf_model', nargs='?', default='lightgcn', help='Downstream Collaborative Filtering model {mf, ngcf, lightgcn, mmgcn, vbpr, hafr, bm3}')   

    # train
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate.')  
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='')  #

    # model
    parser.add_argument('--layers', type=int, default=1, help='Number of item graph conv layers')  
    parser.add_argument('--drop_rate', type=float, default=0.0, help='dropout rate')
    parser.add_argument('--mask_rate', type=float, default=0.1, help='mask_rate')   
    parser.add_argument('--user_cat_rate', type=float, default=2.8, help='user_cat_rate')
    parser.add_argument('--item_cat_rate', type=float, default=0.005, help='item_cat_rate')
    parser.add_argument('--model_cat_rate', type=float, default=0.02, help='model_cat_rate')
    parser.add_argument('--ID_layers', type=int, default=1, help='Number of item graph conv layers')  

    # loss
    parser.add_argument('--aug_mf_rate', type=float, default=0.012, help='user_cat_rate')      # 
    parser.add_argument('--prune_loss_drop_rate', type=float, default=0.71, help='prune_loss_drop_rate')    
    parser.add_argument('--mm_mf_rate', type=float, default=0.0001, help='mm_mf_rate')    
    parser.add_argument('--feat_loss_type', default="sce", type=str, help='feat_loss_type')  #
    parser.add_argument('--att_re_rate', type=float, default=0.00001, help='att_re_rate')      # 
    parser.add_argument("--alpha_l", type=float, default=2, help="`pow`inddex for `sce` loss")
    parser.add_argument('--aug_sample_rate', type=float, default=0.1, help='aug_sample_rate')
    parser.add_argument('--mf_emb_rate', type=float, default=0.0, help='mf_emb_rate')

    return parser.parse_args()
#---LLMAug netflix---------------------------------------------------------------------------------------------------------------------------



# #---LLMAug moivelens---------------------------------------------------------------------------------------------------------------------------
# def parse_args():
#     parser = argparse.ArgumentParser(description="")

#     parser.add_argument('--data_path', nargs='?', default='/home/weiw/Code/LLMs/data/', help='Input data path.')  #/home/weiw/Code/MM/MICRO2Ours/data/       /home/weiw/Datasets/MM/LATTICE/
#     parser.add_argument('--seed', type=int, default=2022, help='Random seed')
#     parser.add_argument('--dataset', nargs='?', default='preprocessed_raw_MovieLens', help='Choose a dataset from {preprocessed_raw_MovieLens, netflix_valid_item}')
#     parser.add_argument('--verbose', type=int, default=5, help='Interval of evaluation.')
#     parser.add_argument('--epoch', type=int, default=1000, help='Number of epoch.')  #default: 1000
#     parser.add_argument('--regs', nargs='?', default='[1e-5,1e-5,1e-2]', help='Regularizations.')  #default: '[1e-5,1e-5,1e-2]'

#     parser.add_argument('--embed_size', type=int, default=64, help='Embedding size.')                     
#     parser.add_argument('--weight_size', nargs='?', default='[64, 64]', help='Output sizes of every layer')  #default: '[64, 64]'
#     parser.add_argument('--topk', type=int, default=10, help='K value of k-NN sparsification')  
#     parser.add_argument('--early_stopping_patience', type=int, default=7, help='') 
#     parser.add_argument('--mess_dropout', nargs='?', default='[0.1, 0.1]', help='Keep probability w.r.t. message dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')

#     parser.add_argument('--sparse', type=int, default=1, help='Sparse or dense adjacency matrix')   
#     parser.add_argument('--debug', action='store_true')  
#     parser.add_argument('--norm_type', nargs='?', default='sym', help='Adjacency matrix normalization operation') 
#     parser.add_argument('--gpu_id', type=int, default=0, help='GPU id')
#     parser.add_argument('--Ks', nargs='?', default='[10, 20, 50]', help='K value of ndcg/recall @ k')
#     parser.add_argument('--test_flag', nargs='?', default='part', help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')


#     parser.add_argument('--sc', type=float, default=1.0, help='GCN self connection')
#     parser.add_argument('--model_cat_rate', type=float, default=0.02, help='model_cat_rate')
#     parser.add_argument('--feat_reg_decay', default=1e-5, type=float, help='feat_reg_decay') 
#     parser.add_argument('--cf_model', nargs='?', default='lightgcn', help='Downstream Collaborative Filtering model {mf, ngcf, lightgcn, mmgcn, vbpr, hafr, bm3}')   


#     parser.add_argument('--isload', default=False , type=bool, help='whether load model')  #
#     parser.add_argument('--isJustTest', default=False , type=bool, help='whether load model')
#     parser.add_argument('--loadModelPath', default='/home/ww/Code/work3/BSTRec/Model/retailrocket/for_meta_hidden_dim_dim__8_retailrocket_2021_07_10__18_35_32_lr_0.0003_reg_0.01_batch_size_1024_gnn_layer_[16,16,16].pth', type=str, help='loadModelPath')
#     parser.add_argument('--title', default="try_to_draw_line", type=str, help='')  #


#     # train
#     parser.add_argument('--batch_size', type=int, default=1024, help='Batch size.')
#     parser.add_argument('--lr', type=float, default=0.0075, help='Learning rate.')  # lr:0.000011, 0.0005
#     parser.add_argument('--weight_decay', default=1e-4, type=float, help='')  #

#     # model
#     parser.add_argument('--layers', type=int, default=1, help='Number of item graph conv layers')  
#     parser.add_argument('--drop_rate', type=float, default=0.3, help='dropout rate')
#     parser.add_argument('--user_cat_rate', type=float, default=0.72, help='user_cat_rate')      # 2.3
#     parser.add_argument('--item_cat_rate', type=float, default=0.003, help='item_cat_rate')    # 0.005
#     parser.add_argument('--mask_rate', type=float, default=0.1, help='mask_rate')   
#     parser.add_argument('--ID_layers', type=int, default=1, help='Number of item graph conv layers')  

#     # loss
#     parser.add_argument('--aug_mf_rate', type=float, default=0.012, help='aug_mf_rate')      # 
#     parser.add_argument('--prune_loss_drop_rate', type=float, default=0.4, help='prune_loss_drop_rate')   # to tune 
#     parser.add_argument('--mm_mf_rate', type=float, default=0, help='mm_mf_rate')      # 
#     parser.add_argument('--feat_loss_type', default="sce", type=str, help='feat_loss_type')  # to tune
#     parser.add_argument('--att_re_rate', type=float, default=0.00001, help='att_re_rate')      # 
#     parser.add_argument("--alpha_l", type=float, default=2, help="`pow`inddex for `sce` loss")
#     parser.add_argument('--aug_sample_rate', type=float, default=0.0036, help='aug_sample_rate')  # 0.1


#     return parser.parse_args()
# #---LLMAug movielens---------------------------------------------------------------------------------------------------------------------------
