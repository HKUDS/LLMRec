
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
import tqdm
import requests

file_path = ""
max_threads = 5
cnt = 0 

# MovieLens
def construct_prompting(item_attribute, item_list, candidate_list): 
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
    # make candidates
    candidate_string = "Candidates:\n" 
    for index in candidate_list:
        title = item_attribute['title'][index.item()]
        genre = item_attribute['genre'][index.item()]
        candidate_string += "["
        candidate_string += str(index.item())
        candidate_string += "] "
        candidate_string += title + ", "
        candidate_string += genre + "\n"
    # output format
    output_format = "Please output the index of user\'s favorite and least favorite movie only from candidate, but not user history. Please get the index from candidate, at the beginning of each line.\nOutput format:\nTwo numbers separated by '::'. Nothing else.Plese just give the index of candicates, remove [] (just output the digital value), please do not output other thing else, do not give reasoning.\n\n"
    # make prompt
    prompt = "You are a movie recommendation system and required to recommend user with movies based on user history that each movie with title(same topic/doctor), year(similar years), genre(similar genre).\n"
    prompt += history_string
    prompt += candidate_string
    prompt += output_format
    return prompt

# # Netflix
# def construct_prompting(item_attribute, item_list, candidate_list): 
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
#     # make candidates
#     candidate_string = "Candidates:\n" 
#     for index in candidate_list:
#         year = item_attribute['year'][index.item()]
#         title = item_attribute['title'][index.item()]
#         candidate_string += "["
#         candidate_string += str(index.item())
#         candidate_string += "] "
#         candidate_string += str(year) + ", "
#         candidate_string += title + "\n"
#     # output format
#     output_format = "Please output the index of user\'s favorite and least favorite movie only from candidate, but not user history. Please get the index from candidate, at the beginning of each line.\nOutput format:\nTwo numbers separated by '::'. Nothing else.Plese just give the index of candicates, remove [] (just output the digital value), please do not output other thing else, do not give reasoning.\n\n"
#     # make prompt
#     # prompt = "You are a movie recommendation system and required to recommend user with movies based on user history that each movie with title(same topic/doctor), year(similar years), genre(similar genre).\n"
#     prompt = ""
#     prompt += history_string
#     prompt += candidate_string
#     prompt += output_format
#     return prompt

### read candidate 
candidate_indices = pickle.load(open(file_path + 'candidate_indices','rb'))
candidate_indices_dict = {}
for index in range(candidate_indices.shape[0]):
    candidate_indices_dict[index] = candidate_indices[index]
### read adjacency_list
adjacency_list_dict = {}
train_mat = pickle.load(open(file_path + 'train_mat','rb'))
for index in range(train_mat.shape[0]):
    data_x, data_y = train_mat[index].nonzero()
    adjacency_list_dict[index] = data_y
### read item_attribute
toy_item_attribute = pd.read_csv(file_path + 'item_attribute.csv', names=['id','title', 'genre'])
### write augmented dict
augmented_sample_dict = {}
if os.path.exists(file_path + "augmented_sample_dict"): 
    print(f"The file augmented_sample_dict exists.")
    augmented_sample_dict = pickle.load(open(file_path + 'augmented_sample_dict','rb')) 
else:
    print(f"The file augmented_sample_dict does not exist.")
    pickle.dump(augmented_sample_dict, open(file_path + 'augmented_sample_dict','wb')) 
#defining the file reading function
def file_reading():
    augmented_attribute_dict = pickle.load(open(file_path + 'augmented_sample_dict','rb')) 
    return augmented_attribute_dict

# baidu
def LLM_request(toy_item_attribute, adjacency_list_dict, candidate_indices_dict, index, model_type, augmented_sample_dict):

    try:
        #list of augmented samples
        augmented_sample_dict = file_reading()
    except pickle.UnpicklingError as e:
        print("Error occurred while unpickling:", e) 
        LLM_request(toy_item_attribute, adjacency_list_dict, candidate_indices_dict, index, model_type, augmented_sample_dict)
    if index in augmented_sample_dict:
        return 0
    else:
        try: 
            print(f"{index}")
            # make prompting
            prompt = construct_prompting(toy_item_attribute, adjacency_list_dict[index], candidate_indices_dict[index])
            url = "http://llms-se.baidu-int.com:8200/chat/completions"
            headers={
                # "Content-Type": "application/json",
                "Authorization": "Bearer your key"
            
            }
            #defining the parameters
            params={
                "model": model_type,
                "messages": [{"role": "user", "content": prompt}],
                "temperature":0.6,
                "max_tokens": 1000,
                "stream": False, 
                "top_p": 0.1
            }
            # params = {
            response = requests.post(url=url, headers=headers,json=params)
            message = response.json()
            #response
            content = message['choices'][0]['message']['content']
            print(f"content: {content}, model_type: {model_type}")
            #get the samples
            samples = content.split("::")
            pos_sample = int(samples[0])
            neg_sample = int(samples[1])
            augmented_sample_dict[index] = {}
            augmented_sample_dict[index][0] = pos_sample
            augmented_sample_dict[index][1] = neg_sample
            #generate the pickle file
            pickle.dump(augmented_sample_dict, open(file_path + 'augmented_sample_dict','wb'))

        # except ValueError as e:
        except requests.exceptions.RequestException as e:
            print("An HTTP error occurred:", str(e))
            time.sleep(10)
        except ValueError as ve:
            print("An error occurred while parsing the response:", str(ve))
            time.sleep(10)
            LLM_request(toy_item_attribute, adjacency_list_dict, candidate_indices_dict, index, "gpt-3.5-turbo-0613", augmented_sample_dict)
        except KeyError as ke:
            print("An error occurred while accessing the response:", str(ke))
            time.sleep(10)
            LLM_request(toy_item_attribute, adjacency_list_dict, candidate_indices_dict, index, "gpt-3.5-turbo-0613", augmented_sample_dict)
        except Exception as ex:
            print("An unknown error occurred:", str(ex))
            time.sleep(10)
        
        return 1





# # chatgpt
# def LLM_request(toy_item_attribute, adjacency_list_dict, candidate_indices_dict, index, model_type, augmented_sample_dict):

#     if index in augmented_sample_dict:
#         print(f"g:{index}")
#         return 0
#     else:
#         try: 
#             print(f"{index}")
#             prompt = construct_prompting(toy_item_attribute, adjacency_list_dict[index], candidate_indices_dict[index])
#             # url = "http://llms-se.baidu-int.com:8200/chat/completions"
#             # url = "https://api.openai.com/v1/completions"
#             url = "https://api.openai.com/v1/chat/completions"

#             headers={
#                 # "Content-Type": "application/json",
#                 # "Authorization": "Bearer your key"
#                
#             }
#             # params={
#             #     "model": model_type,
#             #     "prompt": prompt,
#             #     "max_tokens": 1024,
#             #     "temperature": 0.6,
#             #     "stream": False,
#             # } 

#             params = {
#                 "model": "gpt-3.5-turbo",
#                 "messages": [{"role": "system", "content": "You are a movie recommendation system and required to recommend user with movies based on user history that each movie with title(same topic/doctor), year(similar years), genre(similar genre).\n"}, {"role": "user", "content": prompt}]
#             }

#             response = requests.post(url=url, headers=headers,json=params)
#             message = response.json()

#             content = message['choices'][0]['message']['content']
#             # content = message['choices'][0]['text']
#             print(f"content: {content}, model_type: {model_type}")
#             samples = content.split("::")
#             pos_sample = int(samples[0])
#             neg_sample = int(samples[1])
#             augmented_sample_dict[index] = {}
#             augmented_sample_dict[index][0] = pos_sample
#             augmented_sample_dict[index][1] = neg_sample
#             # pickle.dump(augmented_sample_dict, open('augmented_sample_dict','wb'))
#             # pickle.dump(augmented_sample_dict, open('/Users/weiwei/Documents/Datasets/ml-10m/ml-10M100K/preprocessed_raw_MovieLens/toy_MovieLens1000/augmented_sample_dict','wb'))
#             pickle.dump(augmented_sample_dict, open(file_path + 'augmented_sample_dict','wb'))

#         # # except ValueError as e:
#         # except requests.exceptions.RequestException as e:
#         #     print("An HTTP error occurred:", str(e))
#         #     # time.sleep(40)
#         # except ValueError as ve:
#         #     print("An error occurred while parsing the response:", str(ve))
#         #     # time.sleep(40)
#         #     LLM_request(toy_item_attribute, adjacency_list_dict, candidate_indices_dict, index, "gpt-3.5-turbo-0613", augmented_sample_dict)
#         # except KeyError as ke:
#         #     print("An error occurred while accessing the response:", str(ke))
#         #     # time.sleep(40)
#         #     LLM_request(toy_item_attribute, adjacency_list_dict, candidate_indices_dict, index, "gpt-3.5-turbo-0613", augmented_sample_dict)
#         # except Exception as ex:
#         #     print("An unknown error occurred:", str(ex))
#         #     # time.sleep(40)
        
#         # return 1

#         # except ValueError as e:
#         except requests.exceptions.RequestException as e:
#             print("An HTTP error occurred:", str(e))
#             time.sleep(8)
#             # print(content)
#             # error_cnt += 1
#             # if error_cnt==5:
#             #     return 1
#             # LLM_request(toy_item_attribute, indices, "gpt-3.5-turbo-0613", augmented_attribute_dict, error_cnt)
#             LLM_request(toy_item_attribute, adjacency_list_dict, candidate_indices_dict, index, model_type, augmented_sample_dict)
#         except ValueError as ve:
#             print("ValueError error occurred while parsing the response:", str(ve))
#             time.sleep(10)
#             # error_cnt += 1
#             # if error_cnt==5:
#             #     return 1
#             # print(content)
#             # LLM_request(toy_item_attribute, indices, "gpt-3.5-turbo-0613", augmented_attribute_dict, error_cnt)
#             LLM_request(toy_item_attribute, adjacency_list_dict, candidate_indices_dict, index, model_type, augmented_sample_dict)
#         except KeyError as ke:
#             print("KeyError error occurred while accessing the response:", str(ke))
#             time.sleep(10)
#             # error_cnt += 1
#             # if error_cnt==5:
#             #     return 1
#             # print(content)
#             # LLM_request(toy_item_attribute, indices, "gpt-3.5-turbo-0613", augmented_attribute_dict, error_cnt)
#             LLM_request(toy_item_attribute, adjacency_list_dict, candidate_indices_dict, index, model_type, augmented_sample_dict)
#         except IndexError as ke:
#             print("IndexError error occurred while accessing the response:", str(ke))
#             time.sleep(10)
#             # error_cnt += 1
#             # if error_cnt==5:
#             #     return 1
#             # # print(content)
#             # LLM_request(toy_item_attribute, indices, "gpt-3.5-turbo-0613", augmented_attribute_dict)
#             # return 1
#             LLM_request(toy_item_attribute, adjacency_list_dict, candidate_indices_dict, index, model_type, augmented_sample_dict)
#         except EOFError as ke:
#             print("EOFError: : Ran out of input error occurred while accessing the response:", str(ke))
#             time.sleep(10)
#             # error_cnt += 1
#             # if error_cnt==5:
#             #     return 1
#             # print(content)
#             # LLM_request(toy_item_attribute, indices, "gpt-3.5-turbo-0613", augmented_attribute_dict, error_cnt)
#             LLM_request(toy_item_attribute, adjacency_list_dict, candidate_indices_dict, index, model_type, augmented_sample_dict)
#         except Exception as ex:
#             print("An unknown error occurred:", str(ex))
#             time.sleep(10)
#             # error_cnt += 1
#             # if error_cnt==5:
#             #     return 1
#             # print(content)
#             # LLM_request(toy_item_attribute, indices, "gpt-3.5-turbo-0613", augmented_attribute_dict, error_cnt)
#             LLM_request(toy_item_attribute, adjacency_list_dict, candidate_indices_dict, index, model_type, augmented_sample_dict)
#         return 1


for index in range(0, len(adjacency_list_dict)):
    # # make prompting
    re = LLM_request(toy_item_attribute, adjacency_list_dict, candidate_indices_dict, index, "gpt-3.5-turbo", augmented_sample_dict)




