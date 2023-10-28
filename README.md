# LLMRec: Large Language Models with Graph Augmentation for Recommendation

PyTorch implementation for WSDM 2024 paper [LLMRec: Large Language Models with Graph Augmentation for Recommendation](https://llmrec.files.wordpress.com/2023/10/wsdm2024llmrec.pdf).



[Wei Wei](#), [Xubin Ren](https://rxubin.com/), [Jiabin Tang](https://tjb-tech.github.io/), [Qingyong Wang](#), [Lixin Su](#), [Suqi Cheng](#), [Junfeng Wang](#), [Dawei Yin](https://www.yindawei.com/) and [Chao Huang](https://sites.google.com/view/chaoh/home)*.
(*Correspondence)

**[Data Intelligence Lab](https://sites.google.com/view/chaoh/home)@[University of Hong Kong](https://www.hku.hk/)**, Baidu Inc.

<a href='https://llmrec.github.io/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
<a href='https://llmrec.github.io/'><img src='https://img.shields.io/badge/Demo-Page-purple'></a>
<a href='https://llmrec.files.wordpress.com/2023/10/wsdm2024llmrec.pdf'><img src='https://img.shields.io/badge/Paper-PDF-orange'></a> 
[![YouTube](https://badges.aleen42.com/src/youtube.svg)](https://www.youtube.com/channel/UC1wKlPPlP9zKGYk62yR0K_g)


This repository hosts the code, original data and augmented data of **LLMRec**.

-----------

<p align="center">
<img src="./image/llmrec_framework.png" alt="LLMRec" />
</p>

LLMRec is a novel framework that enhances recommenders by applying three simple yet effective LLM-based graph augmentation strategies to recommendation system. LLMRec is to make the most of the content within online platforms (e.g., Netflix, MovieLens) to augment interaction graph by i) reinforcing u-i interactive edges, ii) enhancing item node attributes, and iii) conducting user node profiling, intuitively from the natural language perspective.

-----------

## 🎉 News 📢📢  

- [x] [2023.10.29] 🚀🚀 Release the script for constructing the prompt.

- [x] [2023.10.29] 🚀🚀 Release LLM-augmented textual data(by gpt-3.5-turbo-0613), and LLM-augmented embedding(by text-embedding-ada-002).

- [x] [2023.10.28] 🔥🔥 The full paper of our LLMRec is available at [LLMRec: Large Language Models with Graph Augmentation for Recommendation](https://llmrec.files.wordpress.com/2023/10/wsdm2024llmrec.pdf).

- [x] [2023.10.28] 🚀🚀 Release the code of LLMRec.


-----------

<h2> Dependencies </h2>

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

Specific code execution example on 'netflix':
```
# LLMRec
python ./main.py 
# w/o-u-i
python ./main.py
# w/o-u
python ./main.py
# w/o-i
python ./main.py
# w/o-u&i
python ./main.py
```

-----------


<h2> Datasets </h2>

  ```
  ├─ LLMRec/ 
      ├── data/
        ├── netflix/
        ...
  ```

<h3> Multi-modal Datasets </h3>
🌹🌹 Please cite our paper if you use the 'netflix' dataset~ ❤️  

We collected a multi-modal dataset using the original [Netflix Prize Data](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data) released on the [Kaggle](https://www.kaggle.com/) website. The data format is directly compatible with SOTA baselines for multi-modal recommendation like [MMSSL](https://github.com/HKUDS/MMSSL), [LATTICE](https://github.com/CRIPAC-DIG/LATTICE), [MICRO](https://github.com/CRIPAC-DIG/MICRO), etc. 

 `Textual Modality:` We have released the item information curated from the original dataset in the "item_attribute.csv" file. Additionally, we have incorporated textual information enhanced by LLM into the "augmented_item_attribute_agg.csv" file. (The following three images represent (1) information about Kaggle as described on the Kaggle website, (2) textual information from the original Netflix Prize Data, and (3) textual information augmented by LLMs.)
<div style="display: flex; justify-content: center; align-items: flex-start;">
  <figure style="text-align: center; margin: 10px;">
   <img src="./image/textual_data1.png" alt="Image 1" style="width:270px;height:180px;">
<!--     <figcaption>Textual data in original 'Netflix Prize Data' on Kaggle.</figcaption> -->
  </figure>

  <figure style="text-align: center; margin: 10px;">
    <img src="./image/textual_data2.png" alt="Image 2" style="width:270px;height:180px;">
<!--     <figcaption>Textual data in original 'Netflix Prize Data'.</figcaption> -->
  </figure>

  <figure style="text-align: center; margin: 10px;">
    <img src="./image/textual_data3.png" alt="Image 2" style="width:270px;height:180px;">
<!--     <figcaption>LLM-augmented textual data.</figcaption> -->
  </figure>  
</div>
 
 `Visual Modality:` We have released the visual information obtained from web crawling in the "Netflix_Posters" folder. (The following image displays the poster acquired by web crawling using item information from the Netflix Prize Data.)
 <div style="display: flex; justify-content: center; align-items: flex-start;">
  <figure style="text-align: center; margin: 10px;">
   <img src="./image/visiual_data1.png" alt="Image 1" style="width:690px;height:600px;">
<!--     <figcaption>Textual data in original 'Netflix Prize Data' on Kaggle.</figcaption> -->
  </figure>
</div>
 

<h3> Orinigal Multi-modal Datasets & Augmented Datasets </h3>
 <div style="display: flex; justify-content: center; align-items: flex-start;">
  <figure style="text-align: center; margin: 10px;">
   <img src="./image/datasets.png" alt="Image 1" style="width:690px;height:420px;">
<!--     <figcaption>Textual data in original 'Netflix Prize Data' on Kaggle.</figcaption> -->
  </figure>
</div>


<br>
<p>
🌹🌹 The [MovieLens](https://grouplens.org/datasets/movielens/) dataset is released by GroupLens at the University of Minnesota. We collected 'title', 'year', and 'genre' as the basic item-related textual information, while the visual content was obtained from MovieLens through a URL for each item. We have also made available a pre-processed MovieLens dataset that can be used directly with [LLMRec](https://github.com/HKUDS/LLMRec), [MMSSL](https://github.com/HKUDS/MMSSL), [LATTICE](https://github.com/CRIPAC-DIG/LATTICE), and [MICRO](https://github.com/CRIPAC-DIG/MICRO), eliminating the need for any extra data preprocessing, including (1) original images and text, (2) basic user-item interactions, and multi-modal information, (3) as well as LLM-augmented content.

<h3> Encoding the Multi-modal Content. </h3>

We use [CLIP-ViT](https://huggingface.co/openai/clip-vit-base-patch32) and [Sentence-BERT](https://www.sbert.net/) separately as encoders for visual side information and textual side information.




-----------

<h2> Prompt & Completion Example </h2>


<h4> LLM-based Implicit Feedback Augmentation </h4>
> Prompt 
>> ...

> Completion
>> ...





<h4> LLM-based User Profile Augmentation </h4>
> Prompt 
>> ...

> Completion
>> ...



<h4> LLM-based Item Attributes Augmentation </h4>
> Prompt 
>> ...

> Completion
>> ...


<h2> Candidate Preparing for LLM-based Implicit Feedback Augmentation</h2>

 step 1: select base model such as MMSSL or LATTICE
 
 step 2: obtain user embedding and item embedding
 
 step 3: generate candidate
```
      _, candidate_indices = torch.topk(torch.mm(G_ua_embeddings, G_ia_embeddings.T), k=10)  
      pickle.dump(candidate_indices.cpu(), open('./data/' + args.datasets +  '/candidate_indices','wb'))
```
Example of specific candidate data.
```
In [3]: candidate_indices
Out[3]: 
tensor([[ 9765,  2930,  6646,  ..., 11513, 12747, 13503],
        [ 3665,  8999,  2587,  ...,  1559,  2975,  3759],
        [ 2266,  8999,  1559,  ...,  8639,   465,  8287],
        ...,
        [11905, 10195,  8063,  ..., 12945, 12568, 10428],
        [ 9063,  6736,  6938,  ...,  5526, 12747, 11110],
        [ 9584,  4163,  4154,  ...,  2266,   543,  7610]])

In [4]: candidate_indices.shape
Out[4]: torch.Size([13187, 10])
```





-----------

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

The structure of this code is largely based on [MMSSL](https://github.com/HKUDS/MMSSL), [LATTICE](https://github.com/CRIPAC-DIG/LATTICE), [MICRO](https://github.com/CRIPAC-DIG/MICRO). Thank them for their work.

