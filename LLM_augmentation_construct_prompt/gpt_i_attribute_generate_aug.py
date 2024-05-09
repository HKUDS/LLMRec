
import threading
import openai
import time
import pandas as pd
import pickle
import os
import numpy as np
import torch

# openai.api_key = ""
openai.api_key = ""

import requests

file_path = ""


# # MovieLens
# def construct_prompting(item_attribute, indices): 
#     # pre string
#     pre_string = "You are now a search engines, and required to provide the inquired information of the given movies bellow:\n"
#     # make item list
#     item_list_string = ""
#     for index in indices:
#         title = item_attribute['title'][index]
#         genre = item_attribute['genre'][index]
#         item_list_string += "["
#         item_list_string += str(index)
#         item_list_string += "] "
#         item_list_string += title + ", "
#         item_list_string += genre + "\n"
#     # output format
#     output_format = "The inquired information is : director, country, language.\nAnd please output them in form of: \ndirector::country::language\nplease output only the content in the form above, i.e., director::country::language\n, but no other thing else, no reasoning, no index.\n\n"
#     # make prompt
#     prompt = pre_string + item_list_string + output_format
#     return prompt 

# Netflix
def construct_prompting(item_attribute, indices): 
    # pre string
    pre_string = "You are now a search engines, and required to provide the inquired information of the given movies bellow:\n"
    # make item list
    item_list_string = ""
    for index in indices:
        year = item_attribute['year'][index]
        title = item_attribute['title'][index]
        item_list_string += "["
        item_list_string += str(index)
        item_list_string += "] "
        item_list_string += str(year) + ", "
        item_list_string += title + "\n"
    # output format
    output_format = "The inquired information is : director, country, language.\nAnd please output them in form of: \ndirector::country::language\nplease output only the content in the form above, i.e., director::country::language\n, but no other thing else, no reasoning, no index.\n\n"
    # make prompt
    prompt = pre_string + item_list_string + output_format
    return prompt 


# def file_reading():
#     augmented_attribute_dict = pickle.load(open(file_path + 'augmented_attribute_dict','rb')) 
#     return augmented_attribute_dict

# ## baidu attribute generate
# # error_cnt = 0
# global error_cnt
# error_cnt = 0
# def LLM_request(toy_item_attribute, indices, model_type, augmented_attribute_dict, error_cnt):

#     # try:
#     #     augmented_attribute_dict = file_reading()
#     # except pickle.UnpicklingError as e:
#     #     # Handle the unpickling error
#     #     # time.sleep(0.001)
#     #     # augmented_attribute_dict = file_reading()
#     #     print("Error occurred while unpickling:", e) 
#     #     # return
#     #     LLM_request(toy_item_attribute, indices, model_type, augmented_attribute_dict, error_cnt)

#     if indices[0] in augmented_attribute_dict:
#         return 0
#     else:
#         try: 
#             print(f"{indices}")
#             prompt = construct_prompting(toy_item_attribute, indices)
#             url = "http://llms-se.baidu-int.com:8200/chat/completions"
#             headers={
#                 # "Content-Type": "application/json",
#                 # "Authorization": "Bearer your key"
#          
#             }
#             params={
#                 "model": model_type,
#                 "messages": [{"role": "user", "content": prompt}],
#                 "temperature":1,
#                 "max_tokens": 1000,
#                 "stream": False, 
#                 "top_p": 0.1
#             }

#             response = requests.post(url=url, headers=headers,json=params)
#             message = response.json()

#             content = message['choices'][0]['message']['content']
#             print(f"content: {content}, model_type: {model_type}")

#             rows = content.strip().split("\n")  # Split the content into rows
#             for i,row in enumerate(rows):
#                 elements = row.split("::")  # Split each row into elements using "::" as the delimiter
#                 director = elements[0]
#                 country = elements[1]
#                 language = elements[2]
#                 augmented_attribute_dict[indices[i]] = {}
#                 augmented_attribute_dict[indices[i]][0] = director
#                 augmented_attribute_dict[indices[i]][1] = country
#                 augmented_attribute_dict[indices[i]][2] = language
#             # pickle.dump(augmented_sample_dict, open('augmented_sample_dict','wb'))
#             pickle.dump(augmented_attribute_dict, open(file_path + 'augmented_attribute_dict','wb'))
#             # time.sleep(5)

#             error_cnt = 0
#         # except ValueError as e:
#         except requests.exceptions.RequestException as e:
#             print("An HTTP error occurred:", str(e))
#             time.sleep(25)
#             # print(content)
#             error_cnt += 1
#             if error_cnt==5:
#                 return 1
#             LLM_request(toy_item_attribute, indices, model_type, augmented_attribute_dict, error_cnt)
#         except ValueError as ve:
#             print("ValueError error occurred while parsing the response:", str(ve))
#             time.sleep(25)
#             error_cnt += 1
#             if error_cnt==5:
#                 return 1
#             # print(content)
#             LLM_request(toy_item_attribute, indices, model_type, augmented_attribute_dict, error_cnt)
#         except KeyError as ke:
#             print("KeyError error occurred while accessing the response:", str(ke))
#             time.sleep(25)
#             error_cnt += 1
#             if error_cnt==5:
#                 return 1
#             # print(content)
#             LLM_request(toy_item_attribute, indices, model_type, augmented_attribute_dict, error_cnt)
#         except IndexError as ke:
#             print("IndexError error occurred while accessing the response:", str(ke))
#             time.sleep(25)
#             # error_cnt += 1
#             # if error_cnt==5:
#             #     return 1
#             # # print(content)
#             LLM_request(toy_item_attribute, indices, model_type, augmented_attribute_dict, error_cnt)
#             return 1
#         except Exception as ex:
#             print("An unknown error occurred:", str(ex))
#             time.sleep(25)
#             error_cnt += 1
#             if error_cnt==5:
#                 return 1
#             # print(content)
#             LLM_request(toy_item_attribute, indices, model_type, augmented_attribute_dict, error_cnt)
#         return 1


### chatgpt attribute generate
def LLM_request(toy_item_attribute, indices, model_type, augmented_attribute_dict, error_cnt):
    if indices[0] in augmented_attribute_dict:
        return 0
    else:
        try: 
            print(f"{indices}")
            prompt = construct_prompting(toy_item_attribute, indices)
            url = "https://api.openai.com/v1/completions"
            headers={
                # "Content-Type": "application/json",
                "Authorization": "Bearer your key"
            }
            #define the params
            params={
                "model": "text-davinci-003",
                "prompt": prompt,
                "max_tokens": 1024,
                "temperature": 0.6,
                "stream": False,
            } 
            #send request
            response = requests.post(url=url, headers=headers,json=params)
            message = response.json()
            #get the content
            content = message['choices'][0]['text']
            print(f"content: {content}, model_type: {model_type}")

            rows = content.strip().split("\n")  # Split the content into rows
            for i,row in enumerate(rows):
                elements = row.split("::")  # Split each row into elements using "::" as the delimiter
                director = elements[0]
                country = elements[1]
                language = elements[2]
                augmented_attribute_dict[indices[i]] = {}
                augmented_attribute_dict[indices[i]][0] = director
                augmented_attribute_dict[indices[i]][1] = country
                augmented_attribute_dict[indices[i]][2] = language
            # pickle.dump(augmented_sample_dict, open('augmented_sample_dict','wb'))
            pickle.dump(augmented_attribute_dict, open(file_path + 'augmented_attribute_dict','wb'))
        
        # except ValueError as e:
        except requests.exceptions.RequestException as e:
            print("An HTTP error occurred:", str(e))
            # time.sleep(25)
            # print(content)
            error_cnt += 1
            if error_cnt==5:
                return 1
            LLM_request(toy_item_attribute, indices, "gpt-3.5-turbo-0613", augmented_attribute_dict, error_cnt)
        except ValueError as ve:
            print("ValueError error occurred while parsing the response:", str(ve))
            # time.sleep(25)
            error_cnt += 1
            if error_cnt==5:
                return 1
            # print(content)
            LLM_request(toy_item_attribute, indices, "gpt-3.5-turbo-0613", augmented_attribute_dict, error_cnt)
        except KeyError as ke:
            print("KeyError error occurred while accessing the response:", str(ke))
            # time.sleep(25)
            error_cnt += 1
            if error_cnt==5:
                return 1
            # print(content)
            #request again
            LLM_request(toy_item_attribute, indices, "gpt-3.5-turbo-0613", augmented_attribute_dict, error_cnt)
        except IndexError as ke:
            print("IndexError error occurred while accessing the response:", str(ke))
            # time.sleep(25)
            # error_cnt += 1
            # if error_cnt==5:
            #     return 1
            # # print(content)
            # LLM_request(toy_item_attribute, indices, "gpt-3.5-turbo-0613", augmented_attribute_dict)
            return 1
        except Exception as ex:
            print("An unknown error occurred:", str(ex))
            # time.sleep(25)
            error_cnt += 1
            if error_cnt==5:
                return 1
            # print(content)
            #request again
            LLM_request(toy_item_attribute, indices, "gpt-3.5-turbo-0613", augmented_attribute_dict, error_cnt)
        return 1







### chatgpt attribute embedding
def LLM_request(toy_augmented_item_attribute, indices, model_type, augmented_atttribute_embedding_dict, error_cnt):
    for value in augmented_atttribute_embedding_dict.keys():
        print(value)
        if indices[0] in augmented_atttribute_embedding_dict[value]:
            # return 0
            continue 
        else:
            try: 
                print(f"{indices}")
                #generate prompt
                # prompt = construct_prompting(toy_item_attribute, indices)
                url = "https://api.openai.com/v1/embeddings"
                headers={
                    # "Content-Type": "application/json",
                    "Authorization": "Bearer your key"
                }
                #define the params
                params={
                "model": "text-embedding-ada-002",
                "input": toy_augmented_item_attribute[value][indices].values[0]
                }
                #send request
                response = requests.post(url=url, headers=headers,json=params)
                message = response.json()

                content = message['data'][0]['embedding']
                #set the content to the augmented_atttribute_embedding_dict
                augmented_atttribute_embedding_dict[value][indices[0]] = content
                # pickle.dump(augmented_sample_dict, open('augmented_sample_dict','wb'))
                pickle.dump(augmented_atttribute_embedding_dict, open(file_path + 'augmented_atttribute_embedding_dict','wb'))
            
            # except ValueError as e:
            except requests.exceptions.RequestException as e:
                print("An HTTP error occurred:", str(e))
                time.sleep(5)
                # print(content)
                LLM_request(toy_augmented_item_attribute, indices, "text-embedding-ada-002", augmented_atttribute_embedding_dict, error_cnt)
            except ValueError as ve:
                print("An error occurred while parsing the response:", str(ve))
                time.sleep(5)
                # print(content)
                LLM_request(toy_augmented_item_attribute, indices, "text-embedding-ada-002", augmented_atttribute_embedding_dict, error_cnt)
            except KeyError as ke:
                print("An error occurred while accessing the response:", str(ke))
                time.sleep(5)
                # print(content)
                LLM_request(toy_augmented_item_attribute, indices, "text-embedding-ada-002", augmented_atttribute_embedding_dict, error_cnt)
            except Exception as ex:
                print("An unknown error occurred:", str(ex))
                time.sleep(5)
                # print(content)
                LLM_request(toy_augmented_item_attribute, indices, "text-embedding-ada-002", augmented_atttribute_embedding_dict, error_cnt)
            # return 1









#defining the function to read the file
def file_reading():
    #read the augmented_attribute_dict
    augmented_atttribute_embedding_dict = pickle.load(open(file_path + 'augmented_atttribute_embedding_dict','rb')) 
    return augmented_atttribute_embedding_dict

### baidu attribute embedding
def LLM_request(toy_augmented_item_attribute, indices, model_type, augmented_atttribute_embedding_dict, error_cnt, key, file_name):
    for value in augmented_atttribute_embedding_dict.keys():
        if indices[0] in augmented_atttribute_embedding_dict[value]:
            # return 0
            continue
        else:
            try: 
                print(f"{indices}")
                print(value)

                ### chatgpt #############################################################################################################################
                # prompt = construct_prompting(toy_item_attribute, indices)
                url = "https://api.openai.com/v1/embeddings"
                headers={
                    # "Content-Type": "application/json",
                    "Authorization": "Bearer your key"
                }
                ### chatgpt #############################################################################################################################

                #define the params
                params={
                "model": "text-embedding-ada-002",
                "input": str(toy_augmented_item_attribute[value][indices].values[0])
                }
                response = requests.post(url=url, headers=headers,json=params)
                message = response.json()
                #response
                content = message['data'][0]['embedding']
                #set the content to the augmented_atttribute_embedding_dict
                augmented_atttribute_embedding_dict[value][indices[0]] = content
                pickle.dump(augmented_atttribute_embedding_dict, open(file_path + file_name,'wb'))
            
            # except ValueError as e:
            except requests.exceptions.RequestException as e:
                print("An HTTP error occurred:", str(e))
                time.sleep(5)
                # print(content)
                LLM_request(toy_augmented_item_attribute, indices, "text-embedding-ada-002", augmented_atttribute_embedding_dict, error_cnt, key, file_name)
            except ValueError as ve:
                print("An error occurred while parsing the response:", str(ve))
                time.sleep(5)
                # print(content)
                LLM_request(toy_augmented_item_attribute, indices, "text-embedding-ada-002", augmented_atttribute_embedding_dict, error_cnt, key, file_name)
            except KeyError as ke:
                print("An error occurred while accessing the response:", str(ke))
                time.sleep(5)
                # print(content)
                LLM_request(toy_augmented_item_attribute, indices, "text-embedding-ada-002", augmented_atttribute_embedding_dict, error_cnt, key, file_name)
            except Exception as ex:
                print("An unknown error occurred:", str(ex))
                time.sleep(5)
                # print(content)
                LLM_request(toy_augmented_item_attribute, indices, "text-embedding-ada-002", augmented_atttribute_embedding_dict, error_cnt, key, file_name)
            # return 1




# error_cnt = 0
# ############################# step 1: built item attribute ##########################################################
# ### write augmented dict
# augmented_attribute_dict = {}
# if os.path.exists(file_path + "augmented_attribute_dict"): 
#     print(f"The file augmented_attribute_dict exists.")
#     augmented_attribute_dict = pickle.load(open(file_path + 'augmented_attribute_dict','rb')) 
# else:
#     print(f"The file augmented_attribute_dict does not exist.")
#     pickle.dump(augmented_attribute_dict, open(file_path + 'augmented_attribute_dict','wb'))

# ### read item attribute file
# # toy_item_attribute = pd.read_csv(file_path + 'item_attribute.csv', names=['id','title', 'genre'])
# toy_item_attribute = pd.read_csv(file_path + 'item_attribute.csv', names=['id','year', 'title'])

# for i in range(0, toy_item_attribute.shape[0], 1):
#     batch_start = i
#     batch_end = min(i + 1, toy_item_attribute.shape[0])
#     indices = list(range(batch_start, batch_end))
#     print(f"###i###: {i}")
#     print(f"#######: {indices}")
#     re = LLM_request(toy_item_attribute, indices, "gpt-3.5-turbo-16k", augmented_attribute_dict, error_cnt)
#     # if re:
#     #     time.sleep(0.3)

# # # 
# # for i in range(4189, toy_item_attribute.shape[0], 1):
# #     batch_start = i
# #     batch_end = min(i + 1, toy_item_attribute.shape[0])
# #     indices = list(range(batch_start, batch_end))
# #     print(f"###i###: {i}")
# #     print(f"#######: {indices}")
# #     re = LLM_request(toy_item_attribute, indices, "gpt-4", augmented_attribute_dict, error_cnt)
# #     # # if re:
#     # time.sleep(1)
#     # "gpt-3.5-turbo", "gpt-3.5-turbo-0613", "gpt-3.5-turbo-16k"
# ############################# step 1: built item attribute ##########################################################



# ############################# step 2: generate new csv ##########################################################
# import pandas as pd
# # raw_item_attribute = pd.read_csv(file_path + 'item_attribute.csv', names=['id','title','genre'])
# raw_item_attribute = pd.read_csv(file_path + 'item_attribute_filter.csv', names=['id','year','title'])
# augmented_attribute_dict = pickle.load(open(file_path + 'augmented_attribute_dict','rb'))
# director_list, country_list, language_list = [], [], []
# for i in range(len(augmented_attribute_dict)):
#     director_list.append(augmented_attribute_dict[i][0])
#     country_list.append(augmented_attribute_dict[i][1])
#     language_list.append(augmented_attribute_dict[i][2])
# director_series = pd.Series(director_list)
# country_series = pd.Series(country_list)
# language_series = pd.Series(language_list)
# raw_item_attribute['director'] = director_series
# raw_item_attribute['country'] = country_series
# raw_item_attribute['language'] = language_series
# raw_item_attribute.to_csv(file_path + 'augmented_item_attribute_agg.csv', index=False, header=None)
# ############################# step 2: generate new csv ##########################################################


# ############################# step 3: generate item atttribute embedding ##########################################################
# ### write augmented dict
# # emb_dict_name = ['title_embedding_dict', 'genre_embedding_dict', 'director_embedding_dict', 'country_embedding_dict', 'language_embedding_dict']  # TODO: add total
# emb_dict_name = ['year_embedding_dict', 'title_embedding_dict', 'director_embedding_dict', 'country_embedding_dict', 'language_embedding_dict']  # TODO: add total
# title_embedding_dict, genre_embedding_dict, director_embedding_dict, country_embedding_dict, language_embedding_dict = {}, {}, {}, {}, {}
# # augmented_atttribute_embedding_dict = {'title':title_embedding_dict, 'genre':genre_embedding_dict, 'director':director_embedding_dict, 'country':country_embedding_dict, 'language':language_embedding_dict}
# augmented_atttribute_embedding_dict = {'year':genre_embedding_dict, 'title':title_embedding_dict, 'director':director_embedding_dict, 'country':country_embedding_dict, 'language':language_embedding_dict}

# augmented_atttribute_embedding_dict1 = augmented_atttribute_embedding_dict2 = augmented_atttribute_embedding_dict3 = augmented_atttribute_embedding_dict4 = augmented_atttribute_embedding_dict5 = augmented_atttribute_embedding_dict6 = augmented_atttribute_embedding_dict7 = augmented_atttribute_embedding_dict8 = augmented_atttribute_embedding_dict9 = augmented_atttribute_embedding_dict10 = augmented_atttribute_embedding_dict11 = augmented_atttribute_embedding_dict12 = augmented_atttribute_embedding_dict13 = augmented_atttribute_embedding_dict14 = augmented_atttribute_embedding_dict15 = augmented_atttribute_embedding_dict16 = augmented_atttribute_embedding_dict17 = augmented_atttribute_embedding_dict18 = augmented_atttribute_embedding_dict19 = augmented_atttribute_embedding_dict20 = augmented_atttribute_embedding_dict21 = augmented_atttribute_embedding_dict22 = augmented_atttribute_embedding_dict23 = augmented_atttribute_embedding_dict24 = augmented_atttribute_embedding_dict25 = augmented_atttribute_embedding_dict26 = augmented_atttribute_embedding_dict27 = augmented_atttribute_embedding_dict28 = augmented_atttribute_embedding_dict29 = augmented_atttribute_embedding_dict30 = augmented_atttribute_embedding_dict31 = augmented_atttribute_embedding_dict32 = augmented_atttribute_embedding_dict33 = augmented_atttribute_embedding_dict34 = augmented_atttribute_embedding_dict35 = augmented_atttribute_embedding_dict
# # if os.path.exists(file_path + "augmented_atttribute_embedding_dict"): 
# #     print(f"The file augmented_atttribute_embedding_dict exists.")
# #     augmented_atttribute_embedding_dict = pickle.load(open(file_path + 'augmented_atttribute_embedding_dict','rb')) 
# # else:
# #     print(f"The file augmented_atttribute_embedding_dict does not exist.")
# #     pickle.dump(augmented_atttribute_embedding_dict, open(file_path + 'augmented_atttribute_embedding_dict','wb'))

# file_name = "augmented_atttribute_embedding_dict12"
# if os.path.exists(file_path + file_name): 
#     print(f"The file augmented_atttribute_embedding_dict exists.")
#     augmented_atttribute_embedding_dict = pickle.load(open(file_path + file_name,'rb')) 
# else:
#     print(f"The file augmented_atttribute_embedding_dict does not exist.")
#     pickle.dump(augmented_atttribute_embedding_dict, open(file_path + file_name,'wb'))



# error_cnt=0
# ### read augmented item attribute file
# # toy_augmented_item_attribute = pd.read_csv(file_path + 'augmented_item_attribute_agg.csv', names=['id','title', 'genre', 'director', 'country', 'language'])
# toy_augmented_item_attribute = pd.read_csv(file_path + 'augmented_item_attribute_agg.csv', names=['id', 'year','title', 'director', 'country', 'language'])


# g_key = ""

# for i in range(5500, 6000, 1):
#     batch_start = i
#     batch_end = min(i + 1, toy_augmented_item_attribute.shape[0])
#     indices = list(range(batch_start, batch_end))
#     # print(f"###i###: {i}")
#     print(f"#######: {indices}")
#     LLM_request(toy_augmented_item_attribute, indices, "text-embedding-ada-002", augmented_atttribute_embedding_dict, error_cnt, g_key, file_name)


# # ### get separate embedding matrix
# # import pandas as pd
# # augmented_atttribute_embedding_dict = pickle.load(open(file_path + 'augmented_atttribute_embedding_dict','rb')) 
# # for value in augmented_atttribute_embedding_dict.keys():
# #     augmented_atttribute_embedding_dict[value] = np.array(augmented_atttribute_embedding_dict[value])


# # raw_item_attribute = pd.read_csv(file_path + 'toy_item_attribute.csv', names=['id','title','genre'])


# # augmented_attribute_dict = pickle.load(open('augmented_attribute_dict','rb'))
# # director_list, country_list, language_list = [], [], []
# # for i in range(len(augmented_attribute_dict)):
# #     director_list.append(augmented_attribute_dict[i][0])
# #     country_list.append(augmented_attribute_dict[i][1])
# #     language_list.append(augmented_attribute_dict[i][2])
# # director_series = pd.Series(director_list)
# # country_series = pd.Series(country_list)
# # language_series = pd.Series(language_list)
# # raw_item_attribute['director'] = director_series
# # raw_item_attribute['country'] = country_series
# # raw_item_attribute['language'] = language_series
# # raw_item_attribute.to_csv(file_path + 'toy_augmented_item_attribute.csv', index=False, header=None)
# ############################# step 3: generate item atttribute embedding ##########################################################



# ############################# step 4: get separate embedding matrix ##########################################################
# augmented_total_embed_dict = {'title':[] , 'genre':[], 'director':[], 'country':[], 'language':[]}   
# augmented_atttribute_embedding_dict = pickle.load(open('augmented_atttribute_embedding_dict','rb'))
# for value in augmented_atttribute_embedding_dict.keys():
#     for i in range(len(augmented_atttribute_embedding_dict[value])):
#         augmented_total_embed_dict[value].append(augmented_atttribute_embedding_dict[value][i])   
#     augmented_total_embed_dict[value] = np.array(augmented_total_embed_dict[value])    
# pickle.dump(augmented_total_embed_dict, open(file_path + 'augmented_total_embed_dict','wb'))
# ############################# step 4: get separate embedding matrix ##########################################################



# ############################# step 5: i-i relation struction:  (constructured when start) ##########################################################
# # augmented_total_embed_dict = pickle.load(open(file_path + 'augmented_total_embed_dict','rb'))
# # for value in augmented_atttribute_embedding_dict.keys():
# #     augmented_atttribute_embedding_dict[value] = torch.tensor(augmented_atttribute_embedding_dict[value])
# pass
# ############################# step 5: i-i relation struction:  (constructured when start) ##########################################################



# # ############################# step 6: agg file ##########################################################
# dict_list = []
# for i in range(1,36):
#     tmp_dict = pickle.load(open('augmented_atttribute_embedding_dict'+str(i),'rb'))
#     dict_list.append(tmp_dict)
# total_dict = {'year':{}, 'title':{}, 'director':{}, 'country':{}, 'language':{}}
# for value in dict_list:
#     for key in total_dict.keys():
#         total_dict[key].update(value[key])
# # ############################# step 6: agg file ##########################################################




