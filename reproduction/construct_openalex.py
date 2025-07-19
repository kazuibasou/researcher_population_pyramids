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

def construct_author_data(individual_countries, start_year, end_year):

    work_data = read_pickle_file('openalex_work_data.pickle')
    author_name = read_pickle_file('openalex_author_name.pickle')

    #p = re.compile('[a-zA-Z]+')
    author_data = {}
    n_works = 0
    for w_id in work_data:
        pub_d = str(work_data[w_id]["pub_d"])
        a_ids = [str(a[0]) for a in work_data[w_id]["a_lst"]]
        topic_id = work_data[w_id]["t_id"]
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
                }

            author_data[a_id]["pubs"].append((w_id, pub_d, a_ids, affs, a_lst))
            author_data[a_id]["affs"] += affs

    aff_to_country = read_pickle_file('openalex_affiliation_to_country.pickle')
    author_data_by_country = {country: {} for country in individual_countries}
    n1 = 0
    for a_id in author_data:
        a_country_lst = list(assign_country_to_author(list(author_data[a_id]["affs"]), aff_to_country))

        author_data[a_id]["country"] = a_country_lst
        author_data[a_id].pop("affs")

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
