import os
import collections
import gzip
import shutil
import glob
import pickle, json
from datetime import datetime
import re
from unidecode import unidecode
import unicodedata
from ftfy import fix_text

#from utils import *

#openalex_snapshot_dir = '../snapshot_data/openalex-snapshot/'
data_dir = './data/'

def read_json_file(f_name):

    f_path = data_dir + f_name
    f = open(f_path, mode='r', encoding='utf-8')
    d = json.load(f)
    f.close()

    return d

def read_pickle_file(f_name):

    f_path = data_dir + f_name
    with open(f_path, mode="rb") as f:
        d = pickle.load(f)

    return d

# def construct_author_name():

#     author_data_dir = openalex_snapshot_dir + "data/authors/"
#     dirs = sorted(list(os.listdir(author_data_dir)))

#     author_name = {}

#     for dir in dirs:

#         if dir == ".DS_Store" or dir == "manifest":
#             continue

#         path = author_data_dir + str(dir)
#         files = [f for f in sorted(list(os.listdir(path))) if os.path.isfile(os.path.join(path, f))]
#         for f_name in files:
#             if ".gz" not in f_name[-3:]:
#                 continue

#             print(path + "/" + f_name)

#             with gzip.open(path + "/" + f_name, mode="rt", encoding='utf-8') as f_gzip:
#                 with open(path + "/" + str(f_name[:-3]), mode="w", encoding='utf-8') as f:
#                     shutil.copyfileobj(f_gzip, f)

#             f = open(path + "/" + str(f_name[:-3]), 'r', encoding='utf-8')
#             lines = f.readlines()
#             f.close()
            
#             for line in lines:
#                 data = ''.join(list(line[:-1]))
#                 d = json.loads(data)

#                 a_id = str(str(d["id"]).replace('https://openalex.org/', ''))

#                 if "display_name" in d:
#                     author_name[a_id] = str(d["display_name"]).lower()

#             os.remove(path + "/" + str(f_name[:-3]))

#     with open(data_dir + "openalex_author_name.pickle", mode='wb') as f:
#         pickle.dump(author_name, f)

#     print("Success: construction of author name data.")
#     print("Number of author names:", len(author_name))

#     return

# def construct_affiliation_to_country():

#     affiliation_to_country = {}

#     institution_data_dir = openalex_snapshot_dir + "data/institutions/"
#     dirs = sorted(list(os.listdir(institution_data_dir)))

#     for dir in dirs:

#         files = sorted(list(glob.glob(institution_data_dir + str(dir) + "/*")))
#         for f_name in files:
#             if ".gz" not in f_name[-3:]:
#                 continue
            
#             print(f_name)

#             with gzip.open(f_name, mode="rt", encoding='utf-8') as f_gzip:
#                 with open(str(f_name[:-3]), mode="w", encoding='utf-8') as f:
#                     shutil.copyfileobj(f_gzip, f)

#             f = open(str(f_name[:-3]), 'r', encoding='utf-8')
#             lines = f.readlines()
#             f.close()

#             for line in lines:
#                 data = ''.join(list(line[:-1]))
#                 d = json.loads(data)

#                 inst_id = str(str(d["id"]).replace('https://openalex.org/', ''))

#                 if "country_code" in d:
#                     affiliation_to_country[inst_id] = str(d["country_code"])

#             os.remove(str(f_name[:-3]))

#     with open(data_dir + "openalex_affiliation_to_country.pickle", mode='wb') as f:
#         pickle.dump(affiliation_to_country, f)

#     print("Success: construction of affiliation-to-country data.")
#     print("Number of institutions:", len(affiliation_to_country))

#     return

def assign_country_to_author(affs, aff_to_country):
    a_countries = []
    for aff_id in affs:
        if aff_to_country.get(aff_id, 'None') != 'None':
            a_countries.append(str(aff_to_country[aff_id]))

    if len(a_countries) == 0:
        return []

    c_counter = collections.Counter(a_countries)
    (c, n_c) = c_counter.most_common()[0]
    c_lst = []
    c_lst.append(c)

    for c_ in c_counter:
        if c_ != c and c_counter[c_] == c_counter[c]:
            c_lst.append(c_)

    return c_lst

def name_cleaning(name):
    name = fix_text(name)  # Unicode修復
    name = unidecode(name)  # アクセント除去
    name = re.sub(r"[^a-zA-Z]", "", name)  # 英字以外除去
    return name.title()  # 頭文字だけ大文字に

def split_name_into_words(name):

    return [str(word) for word in list(name.split(' ')) if len(word) > 0]

def extract_first_and_last_names(name, target_country):
    name = str(' '.join(name.split()))

    # a_name = ""
    # p = re.compile('[a-zA-Z]+')
    # for i in range(0, len(name)):
    #     s = str(str(unidecode.unidecode(name[i])).lower())
    #     if p.fullmatch(s) != None:
    #         a_name += str(name[i]).lower()
    #     elif str(name[i]) in {" ", ".", "-"}:
    #         a_name += str(name[i])

    a_name = name_cleaning(name)

    name_words = split_name_into_words(a_name)

    if len(name_words) < 2:
        return []

    first_last_name_pair_lst = []

    last_name = str(name_words[-1])

    if target_country in {'JP', 'CN', 'KR'}:
        first_name = str(name_words[0])
        first_last_name_pair_lst.append((first_name, last_name))

        for i in range(1, len(name_words) - 1):
            first_name += "-" + str(name_words[i])
            first_last_name_pair_lst.append((first_name, last_name))

        first_name = str(name_words[0])

        for i in range(1, len(name_words) - 1):
            first_name += " " + str(name_words[i])
            first_last_name_pair_lst.append((first_name, last_name))
    else:
        first_name = str(name_words[0])
        first_last_name_pair_lst.append((first_name, last_name))

    return first_last_name_pair_lst

def assign_gender_to_author(a_name, a_country, a_pubs, name_to_gender):

    if a_country == 'None':
        return "N/A"

    first_last_name_pair_lst = extract_first_and_last_names(a_name, a_country)

    gender_acc = {
        'female': 0,
        'male': 0,
    }

    pub_ds = sorted([datetime.strptime(str(pub_d), '%Y-%m-%d') for (w_id, pub_d, a_ids) in a_pubs])
    f_y = int(pub_ds[0].year)

    if a_country == 'CN':
        if 2001 <= f_y and f_y <= 2010:
            min_samples = 50
        else:
            min_samples = 10
    else:
        min_samples = 10

    for (first_name, last_name) in first_last_name_pair_lst:
        if len(list(str(first_name.replace(".", " ").replace("-", " ")).split())) < 1:
            continue

        if len(max(list(str(first_name.replace(".", " ").replace("-", " ")).split()), key=len)) < 2:
            continue

        if a_country in {'CN', 'JP', 'KR'}:
            # try local
            if first_name in name_to_gender.get(a_country, {}):
                g = str(name_to_gender[a_country][first_name].get("gender", ""))
                acc = int(name_to_gender[a_country][first_name].get("accuracy", 0))
                num_samples = int(name_to_gender[a_country][first_name].get("samples", 0))
                if g in {'male', 'female'} and acc > gender_acc.get(g, 0) and num_samples >= min_samples:
                    gender_acc[g] = acc
        else:
            # try global
            if first_name in name_to_gender.get('None', {}):
                g = str(name_to_gender['None'][first_name].get("gender", ""))
                acc = int(name_to_gender['None'][first_name].get("accuracy", 0))
                num_samples = int(name_to_gender['None'][first_name].get("samples", 0))
                if g in {'male', 'female'} and acc > gender_acc.get(g, 0) and num_samples >= min_samples:
                    gender_acc[g] = acc

    if gender_acc['male'] == gender_acc['female']:
        return "N/A"

    if a_country == 'CN':
        if f_y <= 1990:
            accuracy_threshold = 90
        elif f_y >= 2011:
            accuracy_threshold = 95
        else:
            accuracy_threshold = 99
    elif a_country == 'JP':
        if f_y <= 1990:
            accuracy_threshold = 99
        else:
            accuracy_threshold = 90
    else:
        accuracy_threshold = 90

    g = max(gender_acc, key=gender_acc.get)
    if gender_acc[g] >= accuracy_threshold:
        return g
    else:
        return "N/A"

def assign_discipline_to_author(a_disciplines):

    if len(a_disciplines) == 0:
        return []

    dsp_counter = collections.Counter(a_disciplines)
    (dsp, n_dsp) = dsp_counter.most_common()[0]
    dsp_lst = []
    dsp_lst.append(dsp)

    for dsp_ in dsp_counter:
        if dsp_ != dsp and dsp_counter[dsp_] == dsp_counter[dsp]:
            dsp_lst.append(dsp_)

    return dsp_lst

def construct_author_data(individual_countries, start_year, end_year):

    work_data = read_pickle_file('openalex_work_data.pickle')
    author_name = read_pickle_file('openalex_author_name.pickle')
    #name_to_gender = read_json_file('openalex_author_name_to_gender.json')
    topic_data = read_pickle_file("openalex_topic_data.pickle")

    #p = re.compile('[a-zA-Z]+')
    author_data = {}
    n_works = 0
    for w_id in work_data:
        pub_d = str(work_data[w_id]["pub_d"])
        a_ids = [str(a[0]) for a in work_data[w_id]["a_lst"]]
        topic_id = work_data[w_id]["t_id"]
        field = topic_data[topic_id]["field"]['display_name']
        a_lst = list(work_data[w_id]["a_lst"])

        pub_y = int(datetime.strptime(str(pub_d), '%Y-%m-%d').year)
        if pub_y < start_year or pub_y > end_year:
            continue

        n_works += 1

        for (a_id, affs) in work_data[w_id]["a_lst"]:

            #a_name = str(str(unidecode.unidecode(str(author_name[a_id]))).lower().replace("-", "").replace(".", "").replace(" ", ""))
            #if p.fullmatch(a_name) == None:
            #    continue

            if a_id not in author_data:
                author_data[a_id] = {
                    "pubs": [],
                    "affs": [],
                    "name": str(author_name[a_id]),
                    "fields": [],
                }

            author_data[a_id]["pubs"].append((w_id, pub_d, a_ids, affs, a_lst))
            author_data[a_id]["affs"] += affs
            author_data[a_id]["fields"].append(field)

    aff_to_country = read_pickle_file('openalex_affiliation_to_country.pickle')
    author_data_by_country = {country: {} for country in individual_countries}
    n1 = 0
    for a_id in author_data:
        a_country_lst = list(assign_country_to_author(list(author_data[a_id]["affs"]), aff_to_country))
        a_dsp_lst = assign_discipline_to_author(list(author_data[a_id]["fields"]))

        author_data[a_id]["country"] = a_country_lst
        author_data[a_id]["discipline"] = a_dsp_lst
        author_data[a_id].pop("affs")
        author_data[a_id].pop("fields")

        if len(a_country_lst) > 0:
            n1 += 1

        for country in a_country_lst:
            if country in individual_countries:
                author_data_by_country[country][a_id] = dict(author_data[a_id])

    #with open(data_dir + "openalex_author_data.pickle", mode='wb') as f:
    #    pickle.dump(author_data, f)

    print("Number of works:", n_works, flush=True)
    print("Number of authors:", len(author_data), flush=True)
    print("Number of authors with country:", n1, flush=True)

    for country in individual_countries:
        with open(data_dir + "openalex_author_data_" + str(country).lower() + ".pickle", mode='wb') as f:
            pickle.dump(author_data_by_country[country], f)

    return

# def construct_work_data(individual_countries, start_year, end_year):
#     openalex_work_data_path = '../data/openalex/openalex_work_data.pickle'
#     with open(openalex_work_data_path, mode="rb") as f:
#         d = pickle.load(f)
#
#     aff_to_country = read_pickle_file('openalex_affiliation_to_country.pickle')
#
#     openalex_work_data = {}
#     openalex_work_data_by_country = {country: {} for country in individual_countries}
#     for w_id in d:
#         openalex_work_data[w_id] = {
#             'pub_d': d[w_id]['pub_d'],
#             't_id': d[w_id]['t_id'],
#             'a_lst': d[w_id]['a_lst'],
#             # 'refs': d[w_id]['refs'],
#         }
#
#         pub_y = int(datetime.strptime(str(d[w_id]['pub_y']), '%Y-%m-%d').year)
#
#         if pub_y < start_year or pub_y > end_year:
#             continue
#
#         countries = set()
#         for (a_id, affs) in openalex_work_data[w_id]["a_lst"]:
#             for aff_id in affs:
#                 if aff_to_country.get(aff_id, 'None') != 'None':
#                     countries.add(str(aff_to_country[aff_id]))
#
#         for country in countries:
#             if country in individual_countries:
#                 openalex_work_data_by_country[country][w_id] = dict(openalex_work_data[w_id])
#
#     #with open("./data/openalex_work_data.pickle", mode='wb') as f:
#     #    pickle.dump(openalex_work_data, f)
#
#     #print("Success: construction of work data.")
#     #print("Number of papers:", len(openalex_work_data))
#
#     for country in individual_countries:
#         with open(data_dir + "openalex_work_data_" + str(country).lower() + ".pickle", mode='wb') as f:
#             pickle.dump(openalex_work_data_by_country[country], f)
#
#     return
