import math
import pickle
import setup
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from collections import defaultdict
from datetime import datetime, timedelta
from itertools import combinations
import numpy as np
import copy
import matplotlib.pyplot as plt
import os
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, roc_curve
import openpyxl
import pandas as pd
import random
import nomquamgender as nqg
import models
from sklearn.utils.class_weight import compute_sample_weight

data_dir = './data/'

def country_selection(arab_league, orbis_test_size, min_ratio_train_to_test, include_middle_name, include_last_name,
                      clearning_special_chars):

    with open("./data/data_sets.pkl", 'rb') as f:
        data_by_country = pickle.load(f)

    name_count_by_country = {c: (len(data_by_country[c][0]), len(data_by_country[c][1])) for c in data_by_country}

    all_country_lst = [_[0] for _ in sorted(list(name_count_by_country.items()), key=lambda x: x[1], reverse=True)]
    country_lst = []
    n, n_ = 0, 0
    arab_minority_group = set()
    arab_name_samples = {}
    for country in all_country_lst:
        f_name_count = len(data_by_country[country][0])
        m_name_count = len(data_by_country[country][1])

        #print(country, "Female:", f_name_count, "Male:", m_name_count)

        if country in arab_league:
            arab_name_samples[country] = (f_name_count, m_name_count)

        if f_name_count < orbis_test_size * min_ratio_train_to_test or m_name_count < orbis_test_size * min_ratio_train_to_test:
            if country in arab_league and f_name_count > orbis_test_size and m_name_count > orbis_test_size:
                arab_minority_group.add(country)
            continue

        n_ += f_name_count + m_name_count
        country_lst.append(country)

    country_lst += list(arab_minority_group)

    # f_name = "./data/complementnb_best_alpha.pickle"
    # with open(f_name, "rb") as f:
    #     nb_alpha = pickle.load(f)

    f_name = "./data/complementnb_prob_threshold_in_orbis.pickle"
    with open(f_name, "rb") as f:
        prob_threshold_orbis = pickle.load(f)

    f_name = "./data/complementnb_prob_threshold_in_wgnd.pickle"
    with open(f_name, "rb") as f:
        prob_threshold_wgnd = pickle.load(f)

    countries_with_low_auc = {'CN', 'IN', 'BD', 'KR'}

    f_path = ("./data/author_gender"
              + "_middle_name_" + str(include_middle_name)
              + "_last_name_" + str(include_last_name)
              + "_special_chars_" + str(clearning_special_chars)
              + ".pkl")
    with open(f_path, mode="rb") as f:
        author_gender = pickle.load(f)

    f_path = ("./data/author_sample_lst"
              + "_middle_name_" + str(include_middle_name)
              + "_last_name_" + str(include_last_name)
              + "_special_chars_" + str(clearning_special_chars)
              + ".pkl")

    with open(f_path, mode="rb") as f:
        (author_name_corpus, author_sample_lst) = pickle.load(f)

    num_authors_by_country = {}
    for country in country_lst:

        n_f, n_m = 0, 0
        for author_id in author_sample_lst[country]:
            g = author_gender.get(author_id, -1)
            if g == 0:
                n_f += 1
            elif g == 1:
                n_m += 1

        num_authors_by_country[country] = [n_f, n_m]

    prob_threshold = {}
    new_country_lst = []
    for country in country_lst:
        p1 = prob_threshold_orbis.get(country, 0.0)
        p2 = prob_threshold_wgnd.get(country, 0.0)
        n_f = num_authors_by_country[country][0]
        n_m = num_authors_by_country[country][1]

        if p1 < 0.9 or p2 < 0.9 or country in countries_with_low_auc or n_f < 1000 or n_m < 1000:
            continue

        prob_threshold[country] = max(p1, p2)
        new_country_lst.append(country)

    country_lst = list(new_country_lst)

    return country_lst

def read_pickle_file(f_name):

    f_path = data_dir + f_name
    with open(f_path, mode="rb") as f:
        d = pickle.load(f)

    return d

def construct_author_sample_lst(individual_countries, include_middle_name, include_last_name,
                                clearning_special_chars):

    author_name_corpus = defaultdict(list)
    author_sample_lst = defaultdict(list)

    for country in individual_countries:
        author_data = read_pickle_file("openalex_author_data_" + str(country).lower() + ".pickle")
        author_ids = sorted(list(author_data.keys()), reverse=False)

        for i in range(0, len(author_ids)):
            author_id = author_ids[i]

            author_name = author_data[author_id]["name"].lower()

            if len(author_name) <= 0:
                continue

            name_words = [str(word) for word in list(author_name.split(' ')) if len(word) > 0]

            if len(name_words) <= 1:
                continue

            if len(name_words) == 2:
                first_name = str(name_words[0])
                middle_name = ""
                last_name = str(name_words[-1])
            elif len(name_words) == 3:
                first_name = str(name_words[0])
                middle_name = str(name_words[1])
                last_name = str(name_words[-1])
            else:
                first_name = str(name_words[0])
                middle_name = ' '.join([str(name_words[k]) for k in range(1, len(name_words)-1)])
                last_name = str(name_words[-1])

            if not include_middle_name:
                middle_name = ""

            if not include_last_name:
                last_name = ""

            name_token = setup.tokenize_name(first_name, middle_name, last_name, clearning_special_chars)
            first_pub_y = min([int(datetime.strptime(pub[1], '%Y-%m-%d').year) for pub in author_data[author_id]["pubs"]])

            author_name_corpus[country].append(name_token)
            author_sample_lst[country].append(author_id)
            #author_sample_lst[country].append((author_id, first_pub_y))

    f_name = ("./data/author_sample_lst"
              + "_middle_name_" + str(include_middle_name)
              + "_last_name_" + str(include_last_name)
              + "_special_chars_" + str(clearning_special_chars)
              + ".pkl")

    with open(f_name, "wb") as f:
        pickle.dump((author_name_corpus, author_sample_lst), f)

    return (author_name_corpus, author_sample_lst)

def gender_assignment_via_ComplementNB(author_name_corpus, author_sample_lst, country_lst,
                      include_middle_name, include_last_name, clearning_special_chars,
                      nb_alpha, prob_threshold):

    author_gender = dict({})

    for country in country_lst:
        if country == 'N/A':
            continue

        f_name = ("./data/" + str(country).lower() + "_train_val_test_sets"
                  + "_middle_name_" + str(include_middle_name)
                  + "_last_name_" + str(include_last_name)
                  + "_special_chars_" + str(clearning_special_chars)
                  + ".pkl")
        with open(f_name, 'rb') as f:
            c_data = pickle.load(f)
            X_train, y_train = c_data[0], c_data[1]

        f_name = "./data/global_feature_transformers.pkl"
        with open(f_name, 'rb') as f:
            (count_vect, tfidf_transformer) = pickle.load(f)

        X_test_counts = count_vect.transform(author_name_corpus[country])
        X_test = tfidf_transformer.transform(X_test_counts)

        sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

        clf = ComplementNB(alpha=nb_alpha[country], force_alpha=True)
        clf.fit(X_train, y_train, sample_weight=sample_weight)

        y_pred_prob = clf.predict_proba(X_test)

        for i in range(0, len(author_sample_lst[country])):

            if y_pred_prob[i, 0] > y_pred_prob[i, 1]:
                g, prob = 0, y_pred_prob[i, 0]
            elif y_pred_prob[i, 0] < y_pred_prob[i, 1]:
                g, prob = 1, y_pred_prob[i, 1]
            else:
                g, prob = -1, 0

            author_id = author_sample_lst[country][i]

            if author_id not in author_gender:
                author_gender[author_id] = []

            author_gender[author_id].append((country, g, prob))

    n_s = 0
    for author_id in author_gender:
        x = sorted(author_gender[author_id], key=lambda x: x[2], reverse=True)
        prob_max = x[0][2]
        g_set = set()
        for (country, g, prob) in x:
            if prob == prob_max and prob >= prob_threshold[country]:
                g_set.add(g)

        if len(g_set) == 1:
            g = list(g_set)[0]
            author_gender[author_id] = g
            if g in {0, 1}:
                n_s += 1
        else:
            author_gender[author_id] = -1

    print("Number of authors:", len(author_gender), "Number of authors with gender:", n_s)

    f_name = ("./data/author_gender"
              + "_middle_name_" + str(include_middle_name)
              + "_last_name_" + str(include_last_name)
              + "_special_chars_" + str(clearning_special_chars)
              + ".pkl")
    with open(f_name, mode='wb') as f:
        pickle.dump(author_gender, f)

    return author_gender


def calc_women_rate_by_publication_year_and_country(country_lst, include_middle_name, include_last_name, clearning_special_chars):

    f_path = ("./data/author_gender"
              + "_middle_name_" + str(include_middle_name)
              + "_last_name_" + str(include_last_name)
              + "_special_chars_" + str(clearning_special_chars)
              + ".pkl")
    with open(f_path, mode="rb") as f:
        author_gender = pickle.load(f)

    openalex_work_data_path = './data/openalex_work_data.pickle'
    with open(openalex_work_data_path, mode="rb") as f:
        work_data = pickle.load(f)

    f_path = './data/openalex_affiliation_to_country.pickle'
    with open(f_path, mode="rb") as f:
        aff_to_country = pickle.load(f)

    female_and_male_authors_by_year = {country: {y: [[], []] for y in range(1950, 2024+1)} for country in country_lst}

    for w_id in work_data:
        pub_d = datetime.strptime(str(work_data[w_id]['pub_d']), '%Y-%m-%d')
        pub_y = int(pub_d.year)

        for (a_id, affs) in work_data[w_id]['a_lst']:
            for aff_id in affs:
                if aff_to_country.get(aff_id, 'None') == 'None':
                    continue

                country = aff_to_country[aff_id]

                if country not in female_and_male_authors_by_year:
                    continue

                if pub_y not in female_and_male_authors_by_year[country]:
                    continue

                g = author_gender.get(a_id, -1)
                if g not in {0, 1}:
                    continue

                female_and_male_authors_by_year[country][pub_y][g].append(a_id)

    num_women_and_men_by_year = {country: {y: [0, 0] for y in range(1950, 2024 + 1)} for country in country_lst}
    for country in country_lst:
        for pub_y in range(1950, 2024+1):
            n_f = len(set(list(female_and_male_authors_by_year[country][pub_y][0])))
            n_m = len(set(list(female_and_male_authors_by_year[country][pub_y][1])))
            num_women_and_men_by_year[country][pub_y][0] = n_f
            num_women_and_men_by_year[country][pub_y][1] = n_m

    f_path = ("./data/num_women_and_men_by_year"
              + "_middle_name_" + str(include_middle_name)
              + "_last_name_" + str(include_last_name)
              + "_special_chars_" + str(clearning_special_chars)
              + ".pkl")
    with open(f_path, mode='wb') as f:
        pickle.dump(num_women_and_men_by_year, f)

    return num_women_and_men_by_year

def calc_survival_probability_for_publication_interval(target_country, include_middle_name,include_last_name,
                                                       clearning_special_chars):

    openalex_author_data_path = './data/openalex_author_data_' + str(target_country).lower() + '.pickle'
    with open(openalex_author_data_path, mode="rb") as f:
        author_data = pickle.load(f)

    f_path = ("./data/author_gender"
              + "_middle_name_" + str(include_middle_name)
              + "_last_name_" + str(include_last_name)
              + "_special_chars_" + str(clearning_special_chars)
              + ".pkl")
    with open(f_path, mode="rb") as f:
        author_gender = pickle.load(f)

    female_interval_lst = []
    male_interval_lst = []

    for a_id in author_data:

        author_country_lst = set(author_data[a_id]["country"])

        if len(author_country_lst) < 1 or target_country not in author_country_lst:
            continue

        # author_discipline_lst = set(author_data[a_id]["discipline"])
        #
        # if len(author_discipline_lst) < 1:
        #     continue

        if a_id not in author_gender:
            continue

        g = author_gender[a_id]
        if g not in {0, 1}:
            continue

        pub_lst = [pub for pub in author_data[a_id]["pubs"] if int(datetime.strptime(str(pub[1]), '%Y-%m-%d').year)]
        pub_d_lst = sorted([datetime.strptime(str(pub[1]), '%Y-%m-%d') for pub in pub_lst], reverse=False)

        if len(pub_d_lst) < 2:
            continue

        # c_l = float((pub_d_lst[-1] - pub_d_lst[0]).days) / 365
        #
        # if c_l <= 1 or c_l >= 40:
        #     continue
        #
        # pub_rate = float(len(pub_lst)) / c_l
        #
        # if pub_rate >= 20:
        #     continue

        for i in range(0, len(pub_d_lst) - 1):
            pub_d = pub_d_lst[i]
            next_pub_d = pub_d_lst[i + 1]
            d_interval = int((next_pub_d - pub_d).days)
            female_interval_lst.append(d_interval)

            if g == 0:
                female_interval_lst.append(d_interval)
            else:
                male_interval_lst.append(d_interval)

    # Female
    max_interval = max([int(interval) for interval in female_interval_lst])
    female_survival_probability = {y: 0.0 for y in range(0, max_interval + 1)}
    for interval in female_interval_lst:
        for y in range(0, int(interval)+1):
            female_survival_probability[y] += 1

    for y in range(0, max_interval + 1):
        female_survival_probability[y] = float(female_survival_probability[y]) / len(female_interval_lst)

    if female_survival_probability[0] != 1:
        print("ERROR: survival probability.")
        exit()

    # Male
    max_interval = max([int(interval) for interval in male_interval_lst])
    male_survival_probability = {y: 0.0 for y in range(0, max_interval + 1)}
    for interval in male_interval_lst:
        for y in range(0, int(interval)+1):
            male_survival_probability[y] += 1

    for y in range(0, max_interval + 1):
        male_survival_probability[y] = float(male_survival_probability[y]) / len(male_interval_lst)

    if female_survival_probability[0] != 1:
        print("ERROR: survival probability.")
        exit()

    return female_survival_probability, male_survival_probability

def calc_threshold_for_publication_interval(female_survival_probability: dict, male_survival_probability: dict,
                                            prob_threshold=0.01):

    # Female
    x = sorted(list(female_survival_probability.items()), key=lambda x: x[1], reverse=True)
    female_threshold = -1
    for (y, p) in x:
        if p < prob_threshold:
            female_threshold = y
            break

    # Male
    x = sorted(list(male_survival_probability.items()), key=lambda x: x[1], reverse=True)
    male_threshold = -1
    for (y, p) in x:
        if p < prob_threshold:
            male_threshold = y
            break

    return [female_threshold, male_threshold]

def calc_productive_people_pyramid(target_country, include_middle_name,include_last_name,
                                   clearning_special_chars, pub_interval_threshold, target_year):

    openalex_author_data_path = './data/openalex_author_data_' + str(target_country).lower() + '.pickle'
    with open(openalex_author_data_path, mode="rb") as f:
        author_data = pickle.load(f)

    f_path = ("./data/author_gender"
              + "_middle_name_" + str(include_middle_name)
              + "_last_name_" + str(include_last_name)
              + "_special_chars_" + str(clearning_special_chars)
              + ".pkl")
    with open(f_path, mode="rb") as f:
        author_gender = pickle.load(f)

    female_productivity = {}
    male_productivity = {}

    max_productivity = 0
    for a_id in author_data:

        author_country_lst = set(author_data[a_id]["country"])

        if len(author_country_lst) < 1 or target_country not in author_country_lst:
            continue

        # author_discipline_lst = set(author_data[a_id]["discipline"])
        #
        # if len(author_discipline_lst) < 1:
        #     continue

        if a_id not in author_gender:
            continue

        g = author_gender[a_id]
        if g not in {0, 1}:
            continue

        pub_lst = sorted([pub for pub in author_data[a_id]["pubs"] if
                          int(datetime.strptime(str(pub[1]), '%Y-%m-%d').year) <= target_year],
                         key=lambda x: datetime.strptime(str(x[1]), '%Y-%m-%d'), reverse=False)
        pub_d_lst = sorted([datetime.strptime(str(pub[1]), '%Y-%m-%d') for pub in pub_lst], reverse=False)

        if len(pub_d_lst) < 2:
            continue

        c_l = float((pub_d_lst[-1] - pub_d_lst[0]).days) / 365

        if c_l <= 1 or c_l >= 40:
            continue

        pub_rate = float(len(pub_lst)) / c_l

        if pub_rate >= 20:
            continue

        valid_pub_lst = []
        valid_pub_lst.append(pub_lst[0])
        for k in range(1, len(pub_lst)):
            pre_date = datetime.strptime(str(valid_pub_lst[-1][1]), '%Y-%m-%d')
            date = datetime.strptime(str(pub_lst[k][1]), '%Y-%m-%d')
            if int((date - pre_date).days) >= pub_interval_threshold[g]:
                valid_pub_lst = []
            valid_pub_lst.append(pub_lst[k])

        if len(valid_pub_lst) <= 0:
            print("ERROR: no valid publications.")

        date_deadline = datetime(year=int(target_year), month=12, day=31) - timedelta(days=pub_interval_threshold[g])

        survival_flag = False
        for pub in valid_pub_lst:
            pub_d = datetime.strptime(str(pub[1]), '%Y-%m-%d')
            if int((pub_d - date_deadline).days) >= 0:
                survival_flag = True

        if survival_flag == False:
            continue

        n = len(valid_pub_lst)
        max_productivity = max(max_productivity, n)

        if g == 0:
            female_productivity[a_id] = n
        else:
            male_productivity[a_id] = n

    female_count = {n: 0.0 for n in range(1, max_productivity+1)}
    male_count = {n: 0.0 for n in range(1, max_productivity+1)}

    # Female
    for a_id in female_productivity:
        n = female_productivity[a_id]
        if n > 0:
            female_count[n] += 1

    # Male
    for a_id in male_productivity:
        n = male_productivity[a_id]
        if n > 0:
            male_count[n] += 1

    return female_productivity, male_productivity, female_count, male_count

def calc_future_productive_people_pyramid(target_country, include_middle_name,include_last_name,
                                          clearning_special_chars, pub_interval_threshold, base_year, target_year):

    female_productivity, male_productivity, female_count, male_count = calc_productive_people_pyramid(target_country, include_middle_name,
                                                                                    include_last_name,
                                                                                    clearning_special_chars,
                                                                                    pub_interval_threshold, base_year)
    pre_female_productivity, pre_male_productivity, pre_female_count, pre_male_count = calc_productive_people_pyramid(target_country, include_middle_name,
                                                                                    include_last_name,
                                                                                    clearning_special_chars,
                                                                                    pub_interval_threshold, base_year-1)

    max_female_n = max(set(list(female_productivity.values()) + list(pre_female_productivity.values())))
    female_prob = {n: {n_: 0.0 for n_ in range(0, max_female_n+1)} for n in range(1, max_female_n+1)}
    for a_id in pre_female_productivity:
        n1 = pre_female_productivity[a_id]
        n2 = female_productivity.get(a_id, 0)

        if n1 > 0 or n2 >= 0:
            female_prob[n1][n2] += 1

    for n1 in female_prob:
        norm = sum(list(female_prob[n1].values()))
        if norm > 0:
            female_prob[n1] = {n2: float(female_prob[n1][n2]) / norm for n2 in female_prob[n1]}

    for n in range(1, max_female_n+1):
        norm = np.sum(list(female_prob[n].values()))
        if norm > 0 and math.isclose(norm, 1.0) == False:
            print("ERROR: female probability")
            print(norm)
            exit()
        if norm < 0:
            print("ERROR: female probability")
            print(norm)
            exit()

    female_newcomer_count = {n: 0 for n in range(1, max_female_n+1)}
    for a_id in female_productivity:
        n1 = pre_female_productivity.get(a_id, 0)
        n2 = female_productivity[a_id]

        if n1 == 0 and n2 > 0:
            female_newcomer_count[n2] += 1

    max_male_n = max(set(list(male_productivity.values()) + list(pre_male_productivity.values())))
    male_prob = {n: {n_: 0.0 for n_ in range(0, max_male_n+1)} for n in range(1, max_male_n+1)}
    for a_id in pre_male_productivity:
        n1 = pre_male_productivity[a_id]
        n2 = male_productivity.get(a_id, 0)

        if n1 > 0 or n2 >= 0:
            male_prob[n1][n2] += 1

    for n1 in male_prob:
        norm = sum(list(male_prob[n1].values()))
        if norm > 0:
            male_prob[n1] = {n2: float(male_prob[n1][n2]) / norm for n2 in male_prob[n1]}

    for n in range(1, max_male_n+1):
        norm = np.sum(list(male_prob[n].values()))
        if norm > 0 and math.isclose(norm, 1.0) == False:
            print("ERROR: male probability")
            print(norm)
            exit()
        if norm < 0:
            print("ERROR: male probability")
            print(norm)
            exit()

    male_newcomer_count = {n: 0 for n in range(1, max_male_n+1)}
    for a_id in male_productivity:
        n1 = pre_male_productivity.get(a_id, 0)
        n2 = male_productivity[a_id]

        if n1 == 0 and n2 > 0:
            male_newcomer_count[n2] += 1

    future_female_count = {base_year+i: {n: 0.0 for n in range(1, max_female_n+1)} for i in
                           range(0, target_year - base_year + 1)}

    for a_id in female_productivity:
        n = female_productivity[a_id]
        future_female_count[base_year][n] += 1

    for i in range(1, target_year - base_year + 1):
        for n in range(1, max_female_n+1):
            future_female_count[base_year+i][n] = female_newcomer_count[n]
            for k in range(1, max_female_n+1):
                future_female_count[base_year+i][n] += future_female_count[base_year+i-1][k] * female_prob[k][n]

    future_male_count = {base_year + i: {n: 0.0 for n in range(1, max_male_n + 1)} for i in
                           range(0, target_year - base_year + 1)}

    for a_id in male_productivity:
        n = male_productivity[a_id]
        future_male_count[base_year][n] += 1

    for i in range(1, target_year - base_year + 1):
        for n in range(1, max_male_n + 1):
            future_male_count[base_year + i][n] = male_newcomer_count[n]
            for k in range(1, max_male_n + 1):
                future_male_count[base_year + i][n] += future_male_count[base_year + i - 1][k] * male_prob[k][n]

    return future_female_count, future_male_count, female_prob, male_prob, female_newcomer_count, male_newcomer_count
