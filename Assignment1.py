import numpy as np
import pandas as pd
import nltk

df_correct = pd.read_csv("Correct_cities.csv") 
df_misspelt = pd.read_csv("Misspelt_cities.csv") 

dist_dict = {}
final_list = []

# Iterate over each misspelt-name.
for index in df_misspelt.index:
    misspelt_name = df_misspelt["misspelt_name"][index]
    country = df_misspelt["country"][index]
    df_country_wise = df_correct[df_correct["country"]==country]
    least_dist = 999

    # Iterate over each correct-name belonging to the same country.
    for index2 in df_country_wise.index:
        original_name = df_country_wise["name"][index2]
        id = df_country_wise["id"][index2]

        # Calculate the Levenshtein edit-distance.
        dist = nltk.edit_distance(misspelt_name, original_name) 
        if(dist < least_dist):
            least_dist = dist
            dist_dict["misspelt_name"] = misspelt_name
            dist_dict["country"] = country
            dist_dict["original_name"] = original_name
            dist_dict["id"] = id
    final_list.append(dist_dict)
    print(dist_dict)

# final_list is the array of dictionaries of misspelt-name and correct-name.
print(final_list)   