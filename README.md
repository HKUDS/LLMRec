# LLMRec: Large Language Models with Graph Augmentation for Recommendation

PyTorch implementation for WSDM 2024 paper [LLMRec: Large Language Models with Graph Augmentation for Recommendation]([https://arxiv.org/pdf/2302.10632.pdf](https://llmrec.files.wordpress.com/2023/10/wsdm2024_llmrec_large_language_models_with_graph_augmentation_for_recommendation.pdf)).

<p align="center">
<img src="./llmrec_framework.png" alt="LLMRec" />
</p>

LLMRec is a novel framework that enhances recommenders by applying three simple yet effective LLM-based graph augmentation strategies to recommendation system. LLMRec is to make the most of the content within online platforms (e.g., Netflix, MovieLens) to augment interaction graph by i) reinforcing u-i interactive edges, ii) enhancing item node attributes, and iii) conducting user node profiling, intuitively from the natural language perspective.


<h2>Dependencies </h2>
```
pip install -r requirments.txt
```



<h2>Usage </h2>

<h4>Stage 1: LLM-based Data Augmentation</h4>
```
cd LLMRec/LLM_augmentation/
python ./try_gpt_ui_aug.py
python ./try_gpt_user_profiling.py
python ./try_gpt_i_attribute_generate_aug.py
```

<h4>Stage 2: Recommender training with LLM-augmented Data</h4>
```
cd LLMRec/
python ./main.py --dataset {DATASET}
```
Supported datasets:  `netflix`, `movielens`


<h2> Datasets </h2>

  ```
  ├─ LLMRec/ 
      ├── data/
        ├── netflix/
        ...
  ```
| Dataset       | Netflix |                             | MovieLens                                |
|---------------|---------|-----------------------------|------------------------------------------|
| Graph         | Ori.    | U                           | I                                        |
|               |         | 13187                       | 17366                                    |
|               | Aug.    | E                           | 26374                                    |
| Ori. Sparsity |         | 99.970%                     | 99.915%                                  |
| Att.          | Ori.    | U: None                     | I: year, title                           |
|               | Aug.    | U[1536]                     | age, gender, liked genre, disliked genre |
|               |         | I[1536]                     | director, country, language              |
| Modality      |         | Textual[768], Visiual [512] | Textual [768], Visiual [512]             |



```
# part of data preprocessing
# #----json2mat--------------------------------------------------------------------------------------------------
import json
from scipy.sparse import csr_matrix
import pickle
import numpy as np
n_user, n_item = 39387, 23033
f = open('/home/weiw/Code/MM/MMSSL/data/clothing/train.json', 'r')  
train = json.load(f)
row, col = [], []
for index, value in enumerate(train.keys()):
    for i in range(len(train[value])):
        row.append(int(value))
        col.append(train[value][i])
data = np.ones(len(row))
train_mat = csr_matrix((data, (row, col)), shape=(n_user, n_item))
pickle.dump(train_mat, open('./train_mat', 'wb'))  
# # ----json2mat--------------------------------------------------------------------------------------------------


# ----mat2json--------------------------------------------------------------------------------------------------
# train_mat = pickle.load(open('./train_mat', 'rb'))
test_mat = pickle.load(open('./test_mat', 'rb'))
# val_mat = pickle.load(open('./val_mat', 'rb'))

# total_mat = train_mat + test_mat + val_mat
total_mat =test_mat

# total_mat = pickle.load(open('./new_mat','rb'))
# total_mat = pickle.load(open('./new_mat','rb'))
total_array = total_mat.toarray()
total_dict = {}

for i in range(total_array.shape[0]):
    total_dict[str(i)] = [index for index, value in enumerate(total_array[i]) if value!=0]

new_total_dict = {}

for i in range(len(total_dict)):
    # if len(total_dict[str(i)])>1:
    new_total_dict[str(i)]=total_dict[str(i)]

# train_dict, test_dict = {}, {}

# for i in range(len(new_total_dict)):
#     train_dict[str(i)] = total_dict[str(i)][:-1]
#     test_dict[str(i)] = [total_dict[str(i)][-1]]

# train_json_str = json.dumps(train_dict)
test_json_str = json.dumps(new_total_dict)

# with open('./new_train.json', 'w') as json_file:
# # with open('./new_train_json', 'w') as json_file:
#     json_file.write(train_json_str)
with open('./test.json', 'w') as test_file:
# with open('./new_test_json', 'w') as test_file:
    test_file.write(test_json_str)
# ----mat2json--------------------------------------------------------------------------------------------------
```


<h2> Experimental Results </h2>


<h1> Citing </h1>

If you find this work helpful to your research, please kindly consider citing our paper.


```
@inproceedings{wei2023llmrec,
  title={LLMRec: Large Language Models with Graph Augmentation for Recommendation},
  author={Wei, Wei and Ren, Xubin and Tang, Jaibin and Wang, Qingyong and Su, Lixin and Cheng, Suqi and Wang, Junfeng and Yin, Dawei and Huang, Chao},
  journal={arXiv preprint arXiv:2308.05697},
  year={2023}
}
```



## Acknowledgement

The structure of this code is largely based on [MMSSL](https://github.com/HKUDS/MMSSL), [LATTICE](https://github.com/CRIPAC-DIG/LATTICE). Thank them for their work.

