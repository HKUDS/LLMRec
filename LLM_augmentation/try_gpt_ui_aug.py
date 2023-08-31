
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






# csv_file_path = "/Users/weiwei/Documents/Datasets/ml-10m/ml-10M100K/preprocessed_raw_MovieLens/toy_MovieLens1000/toy_item_attribute.csv"
# csv_file_path = "/Users/weiwei/Documents/Datasets/ml-10m/ml-10M100K/preprocessed_raw_MovieLens/item_attribute.csv"
file_path = "/Users/weiwei/Documents/Datasets/ml-10m/ml-10M100K/preprocessed_raw_MovieLens/"


# file_path = "/Users/weiwei/Documents/Datasets/netflix/Netflix_LLMAug/preprocessed_raw_Netflix/"
# file_path = "/Users/weiwei/Documents/Datasets/netflix/Netflix_LLMAug/preprocessed_raw_Netflix/netflix_valid_item/"

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
# candidate_values, candidate_indices = torch.topk(torch.mm(G_ua_embeddings, G_ia_embeddings.T), 10, dim=-1)
# pickle.dump(candidate_indices.detach().cpu(), open('/home/ww/FILE_from_ubuntu18/Code/work10/data/toy_MovieLens/candidate_indices','wb'))
# candidate_indices = pickle.load(open('/Users/weiwei/Documents/Datasets/ml-10m/ml-10M100K/preprocessed_raw_MovieLens/toy_MovieLens1000/candidate_indices','rb'))
candidate_indices = pickle.load(open(file_path + 'candidate_indices','rb'))
candidate_indices_dict = {}
for index in range(candidate_indices.shape[0]):
    candidate_indices_dict[index] = candidate_indices[index]
### read adjacency_list
adjacency_list_dict = {}
# train_mat = pickle.load(open('/Users/weiwei/Documents/Datasets/ml-10m/ml-10M100K/preprocessed_raw_MovieLens/toy_MovieLens1000/train_mat','rb'))
train_mat = pickle.load(open(file_path + 'train_mat','rb'))
for index in range(train_mat.shape[0]):
    data_x, data_y = train_mat[index].nonzero()
    adjacency_list_dict[index] = data_y
### read item_attribute
# toy_item_attribute = pd.read_csv(file_path + 'toy_item_attribute.csv', names=['id','title', 'genre'])
toy_item_attribute = pd.read_csv(file_path + 'item_attribute.csv', names=['id','title', 'genre'])
# toy_item_attribute = pd.read_csv(file_path + 'item_attribute.csv', names=['id','year', 'title'])
# toy_item_attribute = pd.read_csv(file_path + 'item_attribute_filter.csv', names=['id','year', 'title'])


# for index in range(toy_item_attribute.shape[0]):
#     print(f"index:{index}, title:{toy_item_attribute['title'][index]}, genre:{toy_item_attribute['genre'][index]}")
### write augmented dict
augmented_sample_dict = {}
# if os.path.exists("/Users/weiwei/Documents/Datasets/ml-10m/ml-10M100K/preprocessed_raw_MovieLens/toy_MovieLens1000/augmented_sample_dict"): 
if os.path.exists(file_path + "augmented_sample_dict"): 
    print(f"The file augmented_sample_dict exists.")
    # augmented_sample_dict = pickle.load(open('/Users/weiwei/Documents/Datasets/ml-10m/ml-10M100K/preprocessed_raw_MovieLens/toy_MovieLens1000/augmented_sample_dict','rb')) 
    augmented_sample_dict = pickle.load(open(file_path + 'augmented_sample_dict','rb')) 
else:
    print(f"The file augmented_sample_dict does not exist.")
    # pickle.dump(augmented_sample_dict, open('/Users/weiwei/Documents/Datasets/ml-10m/ml-10M100K/preprocessed_raw_MovieLens/toy_MovieLens1000/augmented_sample_dict','wb'))
    pickle.dump(augmented_sample_dict, open(file_path + 'augmented_sample_dict','wb')) 
# for index,value in enumerate(adjacency_list_dict.keys()):
#     # make prompting
#     prompt = construct_prompting(toy_item_attribute, adjacency_list_dict[index], candidate_indices_dict[index])



def file_reading():
    augmented_attribute_dict = pickle.load(open(file_path + 'augmented_sample_dict','rb')) 
    return augmented_attribute_dict

# baidu
def LLM_request(toy_item_attribute, adjacency_list_dict, candidate_indices_dict, index, model_type, augmented_sample_dict):

    try:
        augmented_sample_dict = file_reading()
    except pickle.UnpicklingError as e:
        # Handle the unpickling error
        # time.sleep(0.001)
        # augmented_attribute_dict = file_reading()
        print("Error occurred while unpickling:", e) 
        # return
        LLM_request(toy_item_attribute, adjacency_list_dict, candidate_indices_dict, index, model_type, augmented_sample_dict)

    if index in augmented_sample_dict:
        return 0
    else:
        try: 
            print(f"{index}")
            prompt = construct_prompting(toy_item_attribute, adjacency_list_dict[index], candidate_indices_dict[index])
            url = "http://llms-se.baidu-int.com:8200/chat/completions"
            # url = "https://api.openai.com/v1/completions"

            headers={
                # "Content-Type": "application/json",
                # "Authorization": "Bearer "
            
            }
            params={
                "model": model_type,
                "messages": [{"role": "user", "content": prompt}],
                "temperature":0.6,
                "max_tokens": 1000,
                "stream": False, 
                "top_p": 0.1
            }

            response = requests.post(url=url, headers=headers,json=params)
            message = response.json()

            content = message['choices'][0]['message']['content']
            print(f"content: {content}, model_type: {model_type}")
            samples = content.split("::")
            pos_sample = int(samples[0])
            neg_sample = int(samples[1])
            augmented_sample_dict[index] = {}
            augmented_sample_dict[index][0] = pos_sample
            augmented_sample_dict[index][1] = neg_sample
            # pickle.dump(augmented_sample_dict, open('augmented_sample_dict','wb'))
            # pickle.dump(augmented_sample_dict, open('/Users/weiwei/Documents/Datasets/ml-10m/ml-10M100K/preprocessed_raw_MovieLens/toy_MovieLens1000/augmented_sample_dict','wb'))
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
#                 # "Authorization": "Bearer "
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









# max_threads = 4
# with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
#     # Submit tasks to the executor
#     # future_results = [executor.submit(process_row, row) for row in reader]
#     future_results = [executor.map(LLM_request, [toy_item_attribute, adjacency_list_dict, candidate_indices_dict, index, "gpt-3.5-turbo-0613", augmented_sample_dict]) for index,value in enumerate(adjacency_list_dict.keys())]

#     # # executor.map(my_function, args_list)
#     # # Wait for all tasks to complete
#     # concurrent.futures.wait(future_results)









for index in range(0, len(adjacency_list_dict)):
    # # make prompting
    re = LLM_request(toy_item_attribute, adjacency_list_dict, candidate_indices_dict, index, "gpt-3.5-turbo", augmented_sample_dict)
    # if re:
    # time.sleep(0.008)
# "gpt-3.5-turbo", "gpt-3.5-turbo-0613", "gpt-3.5-turbo-16k", "claude", "vicuna-13b"

# def process_row(row):
#     # Extract necessary information from the row
#     title = row[1]
#     genre = row[2]
#     # Additional parameters can be extracted as needed
#     # Call your custom function with the extracted parameters
#     result = construct_prompting(title, genre)
#     # Process the result as needed
#     return result 
# # Open the CSV file
# with open(csv_file_path, "r") as file:
#     # Create a CSV reader
#     # reader = csv.DictReader(file)
#     reader = csv.reader(file)
#     next(reader)
#     # Create a thread pool executor
#     with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
#         # Submit tasks to the executor
#         # future_results = [executor.submit(process_row, row) for row in reader]

#         executor.map(LLM_request, args_list)
#         # Wait for all tasks to complete
#         concurrent.futures.wait(future_results)

### Direct Recommendation
# 9
# prompt = "You are now a search engines, and required to provide the inquired information of the given movies bellow:\n[1] Iron Man 2, 2010, Action, Adventure, Sci-Fi\n[2] Avengers 4, 2019, Action, Adventure, Drama\n[3] Pirates of the Caribbean: Dead Man's Chest (Pirates of the Caribbean 2), 2006, Action, Adventure, Fantasy\n[4] Pirates of the Caribbean: At World's End (Pirates of the Caribbean 3), 2007, Action, Adventure, Fantasy\n[5] Sadako (assuming you meant the movie featuring the character from The Ring franchise), 2019, Horror\n[6] The Invisible Guest (The Vanished Guest), 2016, Crime, Drama, Mystery\n[7] Guardians of the Galaxy Vol. 3, 2023, Action, Adventure, Comedy, Sci-Fi\n[8] Thor: The Dark World (Thor 2), 2013, Action, Adventure, Fantasy\n[9] Aquaman, 2018, Action, Adventure, Fantasy\n[10] Iron Man 3, 2013, Action, Adventure, Sci-Fi\nThe inquired information is : director, country, language. And please output them in form of {\'director\':director, \'country\':country, \'language\':language}\nplease output only the content above, but no other thing else, no reasoning.\n\n"

# 10
# prompt = "You are a movie recommendation system and required to recommend user with movies based on user history that each movie with title(same topic/doctor), year(similar years), genre(similar genre).\nUser history:\n[121] Vampire Lovers, The (1970), Horror\nCandidates:\n[399] More (1998), Animation|IMAX|Sci-Fi\n[121] Vampire Lovers, The (1970), Horror\n[155] Billabong Odyssey (2003), Documentary\n[155] Billabong Odyssey (2003), Documentary\nGive the single index(the number in the [] at the beginning of each row) of the candidate movie that the user may want to interact with.\nOutput format:\nPlese just give the index of candicates, remove [] (just output the digital value), please do not output other thing else, do not give reasoning.\n\n"

# 7
# last: prompt = "You are a movie recommendation system and required to recommend user with movies based on user history that each movie with title(same topic/doctor), year(similar years), genre(similar genre).\nUser history:\n[280] Best of Times, The (1986), Comedy|Drama\n[297] Prince of Darkness (1987), Fantasy|Horror|Sci-Fi|Thriller\n[321] Year of the Horse (1997), Documentary\n[395] North Dallas Forty (1979), Comedy|Drama\nCandidates:\n[121] Vampire Lovers, The (1970), Horror\n[399] More (1998), Animation|IMAX|Sci-Fi\n[121] Vampire Lovers, The (1970), Horror\n[155] Billabong Odyssey (2003), Documentary\n[155] Billabong Odyssey (2003), Documentary\n[499] Bad Girls (1994), Western\nGive the single index(the number in the [] at the beginning of each row) of the candidate movie that the user may want to interact with.\nOutput format:\nPlese just give the index of candicates, remove [] (just output the digital value), please do not output other thing else, do not give reasoning.\n\n"
# first: prompt = "You are a movie recommendation system and required to recommend user with movies based on user history that each movie with title(same topic/doctor), year(similar years), genre(similar genre).\nUser history:\n[280] Best of Times, The (1986), Comedy|Drama\n[297] Prince of Darkness (1987), Fantasy|Horror|Sci-Fi|Thriller\n[321] Year of the Horse (1997), Documentary\n[395] North Dallas Forty (1979), Comedy|Drama\nCandidates:\n[499] Bad Girls (1994), Western\n[121] Vampire Lovers, The (1970), Horror\n[399] More (1998), Animation|IMAX|Sci-Fi\n[121] Vampire Lovers, The (1970), Horror\n[155] Billabong Odyssey (2003), Documentary\n[155] Billabong Odyssey (2003), Documentary\nGive the single index(the number in the [] at the beginning of each row) of the candidate movie that the user may want to interact with.\nOutput format:\nPlese just give the index of candicates, remove [] (just output the digital value), please do not output other thing else, do not give reasoning.\n\n"
# myself
# prompt = "You are a movie recommendation system and required to recommend user with movies based on user history that each movie with title(same topic/doctor), year(similar years), genre(similar genre).\nUser history:\n[1] Iron Man 2, 2010, Action, Adventure, Sci-Fi\n[2] Avengers 4, 2019, Action, Adventure, Drama\n[3] Pirates of the Caribbean: Dead Man's Chest (Pirates of the Caribbean 2), 2006, Action, Adventure, Fantasy\n[4] Pirates of the Caribbean: At World's End (Pirates of the Caribbean 3), 2007, Action, Adventure, Fantasy\nCandidates:\n[5] Sadako (assuming you meant the movie featuring the character from The Ring franchise), 2019, Horror\n[6] The Invisible Guest (The Vanished Guest), 2016, Crime, Drama, Mystery\n[7] Guardians of the Galaxy Vol. 3, 2023, Action, Adventure, Comedy, Sci-Fi\n[8] Thor: The Dark World (Thor 2), 2013, Action, Adventure, Fantasy\n[9] Aquaman, 2018, Action, Adventure, Fantasy\n[10] Iron Man 3, 2013, Action, Adventure, Sci-Fi\nGive the single index(the number in the [] at the beginning of each row) of the candidate movie that the user may want to interact with.\nOutput format:\nPlese just give the index of candicates, remove [] (just output the digital value), please do not output other thing else, do not give reasoning.\n\n" 
 
### Sampling
# # 9
# prompt = "You are a movie recommendation system and required to recommend user with movies based on user history that each movie with title(same topic/doctor), year(similar years), genre(similar genre).\nUser history:\n[332] Heart and Souls (1993), Comedy|Fantasy\n[364] Men with Brooms (2002), Comedy|Drama|Romance\nCandidates:\n[399] More (1998), Animation|IMAX|Sci-Fi\n[121] Vampire Lovers, The (1970), Horror\n[155] Billabong Odyssey (2003), Documentary\nPlease output the index of user's favorite and least favorite movie only from candidate, but not user history. Please get the index from candidate, at the beginning of each line.\nOutput format:\nTwo numbers separated by '::'. Nothing else. Plese just give the index of candicates, remove [] (just output the digital value), please do not output other thing else, do not give reasoning.\n\n"

# # 10
# prompt = "You are a movie recommendation system and required to recommend user with movies based on user history that each movie with title(same topic/doctor), year(similar years), genre(similar genre).\nUser history:\n[121] Vampire Lovers, The (1970), Horror\nCandidates:\n[399] More (1998), Animation|IMAX|Sci-Fi\n[121] Vampire Lovers, The (1970), Horror\n[155] Billabong Odyssey (2003), Documentary\nPlease output the index of user's favorite and least favorite movie only from candidate, but not user history. Please get the index from candidate, at the beginning of each line.\nOutput format:\nTwo numbers separated by '::'. Nothing else.Plese just give the index of candicates, remove [] (just output the digital value), please do not output other thing else, do not give reasoning.\n\n"

# # 7
# prompt = "You are a movie recommendation system and required to recommend user with movies based on user history that each movie with title(same topic/doctor), year(similar years), genre(similar genre).\nUser history:\n[280] Best of Times, The (1986), Comedy|Drama\n[297] Prince of Darkness (1987), Fantasy|Horror|Sci-Fi|Thriller\n[321] Year of the Horse (1997), Documentary\n[395] North Dallas Forty (1979), Comedy|Drama\nCandidates:\n[121] Vampire Lovers, The (1970), Horror\n[399] More (1998), Animation|IMAX|Sci-Fi\n[121] Vampire Lovers, The (1970), Horror\n[155] Billabong Odyssey (2003), Documentary\n[155] Billabong Odyssey (2003), Documentary\n[499] Bad Girls (1994), Western\nPlease output the index of user's favorite and least favorite movie only from candidate, but not user history. Please get the index from candidate, at the beginning of each line.\nOutput format:\nTwo numbers separated by '::'. Nothing else.Plese just give the index of candicates, remove [] (just output the digital value), please do not output other thing else, do not give reasoning.\n\n"
# prompt = "You are a movie recommendation system and required to recommend user with movies based on user history that each movie with title(same topic/doctor), year(similar years), genre(similar genre).\nUser history:\n[280] Best of Times, The (1986), Comedy|Drama\n[297] Prince of Darkness (1987), Fantasy|Horror|Sci-Fi|Thriller\n[321] Year of the Horse (1997), Documentary\n[395] North Dallas Forty (1979), Comedy|Drama\nCandidates:\n[499] Bad Girls (1994), Western\n[121] Vampire Lovers, The (1970), Horror\n[399] More (1998), Animation|IMAX|Sci-Fi\n[121] Vampire Lovers, The (1970), Horror\n[155] Billabong Odyssey (2003), Documentary\nPlease output the index of user's favorite and least favorite movie only from candidate, but not user history. Please get the index from candidate, at the beginning of each line.\nOutput format:\nTwo numbers separated by '::'. Nothing else.Plese just give the index of candicates, remove [] (just output the digital value), please do not output other thing else, do not give reasoning.\n\n"



# # myself
# prompt = "You are a movie recommendation system and required to recommend user with movies based on user history that each movie with title(same topic/doctor), year(similar years), genre(similar genre).\nUser history:\n[1] Iron Man 2, 2010, Action, Adventure, Sci-Fi\n[2] Avengers 4, 2019, Action, Adventure, Drama\n[3] Pirates of the Caribbean: Dead Man's Chest (Pirates of the Caribbean 2), 2006, Action, Adventure, Fantasy\n[4] Pirates of the Caribbean: At World's End (Pirates of the Caribbean 3), 2007, Action, Adventure, Fantasy\nCandidates:\n[5] Sadako (assuming you meant the movie featuring the character from The Ring franchise), 2019, Horror\n[6] The Invisible Guest (The Vanished Guest), 2016, Crime, Drama, Mystery\n[7] Guardians of the Galaxy Vol. 3, 2023, Action, Adventure, Comedy, Sci-Fi\n[8] Thor: The Dark World (Thor 2), 2013, Action, Adventure, Fantasy\n[9] Aquaman, 2018, Action, Adventure, Fantasy\n[10] Iron Man 3, 2013, Action, Adventure, Sci-Fi\nPlease output the index of user's favorite and least favorite movie only from candidate, but not user history. Please get the index from candidate, at the beginning of each line.\nOutput format:\nTwo numbers separated by '::'. Nothing else.Plese just give the index of candicates, remove [] (just output the digital value), please do not output other thing else, do not give reasoning.\n\n"


# bad model: 'vicuna-7b-v1.1', ''



### user profiling

### item attribute 

# ### useful block ######################################################################################################################
# # model_type_list = ["gpt-3.5-turbo", "gpt-3.5-turbo-0613", "gpt-3.5-turbo-16k", "gpt-4", "gpt-4-0613", "vicuna-7b-v1.1", "vicuna-13b"]
# model_type_list = ["gpt-3.5-turbo", "gpt-3.5-turbo-0613", "gpt-3.5-turbo-16k", "gpt-4", "gpt-4-0613", "vicuna-13b"]
# #Eliminated：, "ernie-bot", "chatglm-6b"
# for model in model_type_list:
#     req(prompt, model)
# ### useful block ######################################################################################################################


# with open('/Users/weiwei/Documents/Datasets/netflix/Netflix_LLMAug/movie_title.txt', 'r') as f:
#     prompts = f.readlines()

# threads = []
# for i in range(5):
#     t = threading.Thread(target=req, args=(prompts[i::5],))
#     threads.append(t)
#     t.start()
#     time.sleep(1)

# for t in threads:
#     t.join()



# with open('prompting.txt', 'r') as f:
#     prompts = f.readlines()

# threads = []
# for prompt in prompts:
#     t = threading.Thread(target=req, args=(prompt,))
#     threads.append(t)
#     t.start()
#     time.sleep(1)

# for t in threads:
#     t.join()





