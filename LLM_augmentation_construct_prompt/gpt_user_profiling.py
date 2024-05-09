
import threading
import openai
import time
import pandas as pd
import csv
import requests
import concurrent.futures
import pickle
import torch
import os
import threading
import time
import numpy as np


openai.api_base = "http://llms-se.baidu-int.com:8200"


import requests


file_path = ""
max_threads = 5
cnt = 0 

# MovieLens
def construct_prompting(item_attribute, item_list): 
    # make history string
    history_string = "User history:\n" 
    for index in item_list:
        title = item_attribute['title'][index]
        genre = item_attribute['genre'][index]
        history_string += "["
        history_string += str(index)
        history_string += "] "
        history_string += title + ", "
        history_string += genre + "\n"
    # output format
    output_format = "Please output the following infomation of user, output format:\n{\'age\':age, \'gender\':gender, \'liked genre\':liked genre, \'disliked genre\':disliked genre, \'liked directors\':liked directors, \'country\':country\, 'language\':language}\nPlease do not fill in \'unknown\', but make an educated guess based on the available information and fill in the specific content.\nplease output only the content in format above, but no other thing else, no reasoning, no analysis, no Chinese. Reiterating once again!! Please only output the content after \"output format: \", and do not include any other content such as introduction or acknowledgments.\n\n"
    # make prompt
    prompt = "You are required to generate user profile based on the history of user, that each movie with title, year, genre.\n"
    prompt += history_string
    prompt += output_format
    return prompt

# # Netflix
# def construct_prompting(item_attribute, item_list): 
#     # make history string
#     history_string = "User history:\n" 
#     for index in item_list:
#         year = item_attribute['year'][index]
#         title = item_attribute['title'][index]
#         history_string += "["
#         history_string += str(index)
#         history_string += "] "
#         history_string += str(year) + ", "
#         history_string += title + "\n"
#     # output format
#     output_format = "Please output the following infomation of user, output format:\n{\'age\':age, \'gender\':gender, \'liked genre\':liked genre, \'disliked genre\':disliked genre, \'liked directors\':liked directors, \'country\':country\, 'language\':language}\nPlease do not fill in \'unknown\', but make an educated guess based on the available information and fill in the specific content.\nplease output only the content in format above, but no other thing else, no reasoning, no analysis, no Chinese. Reiterating once again!! Please only output the content after \"output format: \", and do not include any other content such as introduction or acknowledgments.\n\n"
#     # make prompt
#     prompt = "You are required to generate user profile based on the history of user, that each movie with title, year, genre.\n"
#     prompt += history_string
#     prompt += output_format
#     return prompt


# # embedding
# def LLM_request(augmented_user_profiling_dict, index, model_type, augmented_user_init_embedding):

#     if index in augmented_user_init_embedding:
#         return 0
#     else:
#         try: 
#             # print(f"{index}")
#             # prompt = construct_prompting(augmented_user_init_embedding, index)
#             url = "https://api.openai.com/v1/embeddings"
#             headers={
#                 # "Content-Type": "application/json",
#                 "Authorization": "Bearer your key"

#             }
#             params={
#             "model": "text-embedding-ada-002",
#             "input": augmented_user_profiling_dict[index]
#             }

#             response = requests.post(url=url, headers=headers,json=params)
#             message = response.json()
#             content = message['data'][0]['embedding']
#             # print(content)
#             print(index)

#             augmented_user_init_embedding[index] = np.array(content)
#             # pickle.dump(augmented_sample_dict, open('augmented_sample_dict','wb'))
#             pickle.dump(augmented_user_init_embedding, open(file_path + 'augmented_user_init_embedding','wb'))
        

#         # except ValueError as e:
#         except requests.exceptions.RequestException as e:
#             print("An HTTP error occurred:", str(e))
#             time.sleep(5)
#         except ValueError as ve:
#             print("An error occurred while parsing the response:", str(ve))
#             time.sleep(5)
#             LLM_request(augmented_user_profiling_dict, index, "text-embedding-ada-002", augmented_user_init_embedding)
#         except KeyError as ke:
#             print("An error occurred while accessing the response:", str(ke))
#             time.sleep(5)
#             LLM_request(augmented_user_profiling_dict, index, "text-embedding-ada-002", augmented_user_init_embedding)
#         except Exception as ex:
#             print("An unknown error occurred:", str(ex))
#             time.sleep(5)
        
#         return 1
    

### user profile #################################################################################################################################
def get_gpt_response_w_system(model_type, prompt):
    # global system_prompt
    completion = openai.ChatCompletion.create(
        model=model_type,
        messages=[
            # {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
    )
    response = completion.choices[0].message.content
    # print(response)  
    return response

start_id = 0
g_model_type = "gpt-3.5-turbo-0613" 
# # "claude", "chatglm-6b", "hambuger-13b", "baichuan-7B", "gpt-4", "gpt-4-0613"


# toy_item_attribute, adjacency_list_dict, index, "gpt-4", augmented_user_profiling_dict
#define the function to read the file
def file_reading():
    #read the file
    augmented_user_profiling_dict = pickle.load(open(file_path + 'augmented_user_profiling_dict','rb')) 
    return augmented_user_profiling_dict
## baidu user profile generate
def LLM_request(toy_item_attribute, adjacency_list_dict, index, model_type, _, error_cnt):    
    # try:
    #     augmented_user_profiling_dict = file_reading()
    # except pickle.UnpicklingError as e:
    #     # Handle the unpickling error
    #     # time.sleep(0.001)
    #     augmented_user_profiling_dict = file_reading()
    #     print("Error occurred while unpickling:", e)  

    # if index in augmented_user_profiling_dict:
    #     return 0
    # else:
        # try: 
    print(f"{index}")
    #send the request
    prompt = construct_prompting(toy_item_attribute, adjacency_list_dict[index])
    url = "http://llms-se.baidu-int.com:8200/chat/completions"
    headers={
        # "Authorization": "Bearer your key"
    }
    #define the parameters and prompt
    prompt = "Please output the following infomation of user, output format:\n{\'age\':age, \'gender\':gender, \'liked genre\':liked genre, \'disliked genre\':disliked genre, \'liked directors\':liked directors, \'country\':country\, 'language\':language}\nPlease do not fill in \'unknown\', but make an educated guess based on the available information and fill in the specific content.\nplease output only the content in format above, but no other thing else, no reasoning, no analysis, no Chinese. Reiterating once again!! Please only output the content after \"output format: \", and do not include any other content such as introduction or acknowledgments.\n\n" + "User history:\n" + "[332]" + "title: Heart and Souls (1993), " + "genre: Comedy|Fantasy\n" + "[364]" + "title: Men with Brooms (2002), " + "genre: Comedy|Drama|Romance\n" + "You are required to generate user profile based on the history of user, that each movie with title, year, genre.\n"
    params={
        "model": model_type,
        "messages": [{"role": "user", "content": prompt}],
        "temperature":0.8,
        "max_tokens": 1000,
        "stream": False, 
        "top_p": 0.1
    }
    #send the request
    response = requests.post(url=url, headers=headers,json=params)
    message = response.json()
    content = message['choices'][0]['message']['content']
    # content = get_gpt_response_w_system(model_type, prompt)

    print(f"content: {content}, model_type: {model_type}")

    #     augmented_user_profiling_dict[index] = content
    #     pickle.dump(augmented_user_profiling_dict, open(file_path + 'augmented_user_profiling_dict','wb'))
    #     error_cnt = 0
    #     time.sleep(8)
    # # # except ValueError as e:
    # # except requests.exceptions.RequestException as e:
    # #     print("An HTTP error occurred:", str(e))
    # #     time.sleep(25)
    # #     # print(content)
    # #     LLM_request(toy_item_attribute, adjacency_list_dict, index, "gpt-4", augmented_user_profiling_dict)
    # # except ValueError as ve:
    # #     print("An error occurred while parsing the response:", str(ve))
    # #     time.sleep(25)
    # #     # print(content)
    # #     LLM_request(toy_item_attribute, adjacency_list_dict, index, "gpt-4", augmented_user_profiling_dict)
    # # except KeyError as ke:
    # #     print("An error occurred while accessing the response:", str(ke))
    # #     time.sleep(25)
    # #     # print(content)
    # #     LLM_request(toy_item_attribute, adjacency_list_dict, index, "gpt-4", augmented_user_profiling_dict)
    # # except Exception as ex:
    # #     print("An unknown error occurred:", str(ex))
    # #     time.sleep(25)
    # #     # print(content)
    # #     LLM_request(toy_item_attribute, adjacency_list_dict, index, "gpt-4", augmented_user_profiling_dict)
    # # return 1
    
    # # except ValueError as e:
    # except requests.exceptions.RequestException as e:
    #     print("An HTTP error occurred:", str(e))
    #     time.sleep(5)
    #     # print(content)
    #     # error_cnt += 1
    #     # if error_cnt==5:
    #     #     return 1
    #     # LLM_request(toy_item_attribute, indices, "gpt-3.5-turbo-0613", augmented_attribute_dict, error_cnt)
    #     LLM_request(toy_item_attribute, adjacency_list_dict, index, model_type, augmented_user_profiling_dict, error_cnt)
    # except ValueError as ve:
    #     print("ValueError error occurred while parsing the response:", str(ve))
    #     time.sleep(5)
    #     # error_cnt += 1
    #     # if error_cnt==5:
    #     #     return 1
    #     # print(content)
    #     # LLM_request(toy_item_attribute, indices, "gpt-3.5-turbo-0613", augmented_attribute_dict, error_cnt)
    #     LLM_request(toy_item_attribute, adjacency_list_dict, index, model_type, augmented_user_profiling_dict, error_cnt)
    # except KeyError as ke:
    #     print("KeyError error occurred while accessing the response:", str(ke))
    #     time.sleep(5)
    #     # error_cnt += 1
    #     # if error_cnt==5:
    #     #     return 1
    #     # print(content)
    #     # LLM_request(toy_item_attribute, indices, "gpt-3.5-turbo-0613", augmented_attribute_dict, error_cnt)
    #     LLM_request(toy_item_attribute, adjacency_list_dict, index, model_type, augmented_user_profiling_dict, error_cnt)
    # except IndexError as ke:
    #     print("IndexError error occurred while accessing the response:", str(ke))
    #     time.sleep(5)
    #     # error_cnt += 1
    #     # if error_cnt==5:
    #     #     return 1
    #     # # print(content)
    #     # LLM_request(toy_item_attribute, indices, "gpt-3.5-turbo-0613", augmented_attribute_dict)
    #     # return 1
    #     LLM_request(toy_item_attribute, adjacency_list_dict, index, model_type, augmented_user_profiling_dict, error_cnt)
    # except EOFError as ke:
    #     print("EOFError: : Ran out of input error occurred while accessing the response:", str(ke))
    #     time.sleep(5)
    #     # error_cnt += 1
    #     # if error_cnt==5:
    #     #     return 1
    #     # print(content)
    #     # LLM_request(toy_item_attribute, indices, "gpt-3.5-turbo-0613", augmented_attribute_dict, error_cnt)
    #     LLM_request(toy_item_attribute, adjacency_list_dict, index, model_type, augmented_user_profiling_dict, error_cnt)
    # except Exception as ex:
    #     print("An unknown error occurred:", str(ex))
    #     time.sleep(5)
    #     # error_cnt += 1
    #     # if error_cnt==5:
    #     #     return 1
    #     # print(content)
    #     # LLM_request(toy_item_attribute, indices, "gpt-3.5-turbo-0613", augmented_attribute_dict, error_cnt)
    #     LLM_request(toy_item_attribute, adjacency_list_dict, index, model_type, augmented_user_profiling_dict, error_cnt)
    # return 1
### user profile #################################################################################################################################




# ## chatgpt user profile generate
# def LLM_request(toy_item_attribute, adjacency_list_dict, index, model_type, augmented_user_profiling_dict, error_cnt):
#     if index in augmented_user_profiling_dict:
#         return 0
#     else:
#         # try: 
#         print(f"{index}")
#         prompt = construct_prompting(toy_item_attribute, adjacency_list_dict[index])
#         url = "http://llms-se.baidu-int.com:8200/chat/completions"
#         headers={
#             # "Content-Type": "application/json",
#             # "Authorization": "Bearer your key"
#           
#         }
#         params={
#             "model": model_type,
#             "messages": [{"role": "user", "content": prompt}],
#             "temperature":0.6,
#             "max_tokens": 1000,
#             "stream": False, 
#             "top_p": 0.1
#         }

#         response = requests.post(url=url, headers=headers,json=params)
#         message = response.json()

#         content = message['choices'][0]['message']['content']
#         print(f"content: {content}, model_type: {model_type}")

#         augmented_user_profiling_dict[index] = content
#         pickle.dump(augmented_user_profiling_dict, open(file_path + 'augmented_user_profiling_dict','wb'))
#         error_cnt = 0
#         # # except ValueError as e:
#         # except requests.exceptions.RequestException as e:
#         #     print("An HTTP error occurred:", str(e))
#         #     time.sleep(25)
#         #     # print(content)
#         #     LLM_request(toy_item_attribute, adjacency_list_dict, index, "gpt-4", augmented_user_profiling_dict)
#         # except ValueError as ve:
#         #     print("An error occurred while parsing the response:", str(ve))
#         #     time.sleep(25)
#         #     # print(content)
#         #     LLM_request(toy_item_attribute, adjacency_list_dict, index, "gpt-4", augmented_user_profiling_dict)
#         # except KeyError as ke:
#         #     print("An error occurred while accessing the response:", str(ke))
#         #     time.sleep(25)
#         #     # print(content)
#         #     LLM_request(toy_item_attribute, adjacency_list_dict, index, "gpt-4", augmented_user_profiling_dict)
#         # except Exception as ex:
#         #     print("An unknown error occurred:", str(ex))
#         #     time.sleep(25)
#         #     # print(content)
#         #     LLM_request(toy_item_attribute, adjacency_list_dict, index, "gpt-4", augmented_user_profiling_dict)
#         # return 1
     
#         # # except ValueError as e:
#         # except requests.exceptions.RequestException as e:
#         #     print("An HTTP error occurred:", str(e))
#         #     # time.sleep(25)
#         #     # print(content)
#         #     error_cnt += 1
#         #     if error_cnt==5:
#         #         return 1
#         #     # LLM_request(toy_item_attribute, indices, "gpt-3.5-turbo-0613", augmented_attribute_dict, error_cnt)
#         #     LLM_request(toy_item_attribute, adjacency_list_dict, index, "gpt-4", augmented_user_profiling_dict, error_cnt)
#         # except ValueError as ve:
#         #     print("ValueError error occurred while parsing the response:", str(ve))
#         #     # time.sleep(25)
#         #     error_cnt += 1
#         #     if error_cnt==5:
#         #         return 1
#         #     # print(content)
#         #     # LLM_request(toy_item_attribute, indices, "gpt-3.5-turbo-0613", augmented_attribute_dict, error_cnt)
#         #     LLM_request(toy_item_attribute, adjacency_list_dict, index, "gpt-4", augmented_user_profiling_dict, error_cnt)
#         # except KeyError as ke:
#         #     print("KeyError error occurred while accessing the response:", str(ke))
#         #     # time.sleep(25)
#         #     error_cnt += 1
#         #     if error_cnt==5:
#         #         return 1
#         #     # print(content)
#         #     # LLM_request(toy_item_attribute, indices, "gpt-3.5-turbo-0613", augmented_attribute_dict, error_cnt)
#         #     LLM_request(toy_item_attribute, adjacency_list_dict, index, "gpt-4", augmented_user_profiling_dict, error_cnt)
#         # except IndexError as ke:
#         #     print("IndexError error occurred while accessing the response:", str(ke))
#         #     # time.sleep(25)
#         #     error_cnt += 1
#         #     if error_cnt==5:
#         #         return 1
#         #     # # print(content)
#         #     # LLM_request(toy_item_attribute, indices, "gpt-3.5-turbo-0613", augmented_attribute_dict)
#         #     # return 1
#         # except Exception as ex:
#         #     print("An unknown error occurred:", str(ex))
#         #     # time.sleep(25)
#         #     # error_cnt += 1
#         #     # if error_cnt==5:
#         #     #     return 1
#         #     # print(content)
#         #     # LLM_request(toy_item_attribute, indices, "gpt-3.5-turbo-0613", augmented_attribute_dict, error_cnt)
#         #     LLM_request(toy_item_attribute, adjacency_list_dict, index, "gpt-4", augmented_user_profiling_dict, error_cnt)
#         # return 1










error_cnt = 0


### step1: generate user profiling ################################################################################## 
### read item_attribute
toy_item_attribute = pd.read_csv(file_path + '/item_attribute.csv', names=['id','title', 'genre'])
### write augmented dict
augmented_user_profiling_dict = {}  
if os.path.exists(file_path + "augmented_user_profiling_dict"): 
    print(f"The file augmented_user_profiling_dict exists.")
    augmented_user_profiling_dict = pickle.load(open(file_path + 'augmented_user_profiling_dict','rb')) 
else:
    print(f"The file augmented_user_profiling_dict does not exist.")
    pickle.dump(augmented_user_profiling_dict, open(file_path + 'augmented_user_profiling_dict','wb'))

### read adjacency_list
adjacency_list_dict = {}
train_mat = pickle.load(open(file_path + 'train_mat','rb'))
for index in range(train_mat.shape[0]):
    data_x, data_y = train_mat[index].nonzero()
    adjacency_list_dict[index] = data_y

for index in range(start_id, len(adjacency_list_dict.keys())):
    print(index)
    # # make prompting
    re = LLM_request(toy_item_attribute, adjacency_list_dict, index, g_model_type, augmented_user_profiling_dict, error_cnt)
# "claude", "chatglm-6b", "hambuger-13b", "baichuan-7B", "gpt-4", "gpt-4-0613"
### step1: generate user profiling ################################################################################## 





# ### step2: generate user embedding ################################################################################## 

# ### read user_profile
# augmented_user_profiling_dict = pickle.load(open(file_path + 'augmented_user_profiling_dict','rb'))
# ### write augmented_user_init_embedding
# augmented_user_init_embedding = {}  
# if os.path.exists(file_path + "augmented_user_init_embedding"): 
#     print(f"The file augmented_user_init_embedding exists.")
#     augmented_user_init_embedding = pickle.load(open(file_path + 'augmented_user_init_embedding','rb')) 
# else:
#     print(f"The file augmented_user_init_embedding does not exist.")
#     pickle.dump(augmented_user_init_embedding, open(file_path + 'augmented_user_init_embedding','wb'))

# for index,value in enumerate(augmented_user_profiling_dict.keys()):
#     # # make prompting
#     # prompt = construct_prompting(toy_item_attribute, adjacency_list_dict[index], candidate_indices_dict[index])
#     re = LLM_request(augmented_user_profiling_dict, index, "text-embedding-ada-002", augmented_user_init_embedding)
#     # print(f"{index}")
#     # if re:
#     #     time.sleep(0.5)
# ### step2: generate user embedding ################################################################################## 




# # ### step3: get user embedding ################################################################################## 
# augmented_user_init_embedding = pickle.load(open('/Users/weiwei/Documents/Datasets/ml-10m/ml-10M100K/preprocessed_raw_MovieLens/toy_MovieLens1000/augmented_user_init_embedding','rb'))
# augmented_user_init_embedding_list = []
# for i in range(len(augmented_user_init_embedding)):
#     augmented_user_init_embedding_list.append(augmented_user_init_embedding[i])
# augmented_user_init_embedding_final = np.array(augmented_user_init_embedding_list)
# pickle.dump(augmented_user_init_embedding_final, open('/Users/weiwei/Documents/Datasets/ml-10m/ml-10M100K/preprocessed_raw_MovieLens/toy_MovieLens1000/augmented_user_init_embedding_final','wb'))
# # ### step3: get user embedding ################################################################################## 




# ### clean keys  ########################################################################################################
# In [196]: new_augmented_user_profiling_dict = {}
#      ...: for index,value in enumerate(augmented_user_profiling_dict.keys()):
#      ...:     if type(value) == str:
#      ...:         if int(value.strip("'")) in augmented_user_profiling_dict:
#      ...:             continue
#      ...:         else:
#      ...:             new_augmented_user_profiling_dict[int(value.strip("'"))] = augmented_user_profiling_dict[value]
#      ...:     else:
#      ...:         new_augmented_user_profiling_dict[value] = augmented_user_profiling_dict[value]
# ### clean keys  ########################################################################################################
