import math
import pickle
from datetime import datetime, timedelta
import numpy as np
import json
import nomquamgender as nqg
import collections
import csv

data_dir = './data/'


def read_pickle_file(f_name):

    f_path = data_dir + f_name
    with open(f_path, mode="rb") as f:
        d = pickle.load(f)

    return d


def calc_survival_probability_for_publication_interval(input_data_name):

    f_path = data_dir + input_data_name + '.jsonl'
    with open(f_path) as f:
        author_sample_lst = [json.loads(l) for l in f.readlines()]

    female_interval_lst = []
    male_interval_lst = []

    for author in author_sample_lst:
        g = int(author["gender"])
        if g not in {0, 1}:
            continue

        pub_d_lst = sorted([datetime.strptime(str(pub_d), '%Y-%m-%d') for pub_d in author["pub_date_lst"]],
                           reverse=False)

        if len(pub_d_lst) < 2:
            continue

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


def calc_threshold_for_publication_interval(female_survival_probability: dict, male_survival_probability: dict, survival_prob_threshold):

    # Female
    x = sorted(list(female_survival_probability.items()), key=lambda x: x[1], reverse=True)
    female_threshold = -1
    for (y, p) in x:
        if p < survival_prob_threshold:
            female_threshold = y
            break

    # Male
    x = sorted(list(male_survival_probability.items()), key=lambda x: x[1], reverse=True)
    male_threshold = -1
    for (y, p) in x:
        if p < survival_prob_threshold:
            male_threshold = y
            break

    return [female_threshold, male_threshold]


def calc_researcher_population_pyramid(input_data_name, pub_interval_threshold, target_year):

    f_path = data_dir + input_data_name + '.jsonl'
    with open(f_path) as f:
        author_sample_lst = [json.loads(l) for l in f.readlines()]

    female_productivity = {}
    male_productivity = {}
    max_productivity = 0
    for author in author_sample_lst:
        a_id = str(author["id"])
        g = int(author["gender"])
        if g not in {0, 1}:
            continue

        pub_d_lst_ = [pub_d for pub_d in author["pub_date_lst"]
                      if int(datetime.strptime(str(pub_d), '%Y-%m-%d').year) <= target_year]
        pub_d_lst = sorted([datetime.strptime(str(pub_d), '%Y-%m-%d') for pub_d in pub_d_lst_], reverse=False)

        if len(pub_d_lst) < 2:
            continue

        c_l = float((pub_d_lst[-1] - pub_d_lst[0]).days) / 365

        if c_l <= 1 or c_l >= 40:
            continue

        pub_rate = float(len(pub_d_lst)) / c_l

        if pub_rate >= 20:
            continue

        valid_pub_lst = []
        valid_pub_lst.append(pub_d_lst[0])
        for k in range(1, len(pub_d_lst)):
            pre_date = valid_pub_lst[-1]
            date = pub_d_lst[k]
            if int((date - pre_date).days) >= pub_interval_threshold[g]:
                valid_pub_lst = []
            valid_pub_lst.append(pub_d_lst[k])

        if len(valid_pub_lst) <= 0:
            print("ERROR: no valid publications.")

        date_deadline = datetime(year=int(target_year), month=12, day=31) - timedelta(days=pub_interval_threshold[g])

        survival_flag = False
        for pub_d in valid_pub_lst:
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


def calc_future_researcher_population_pyramid(input_data_name, pub_interval_threshold, base_year, target_year):

    female_productivity, male_productivity, female_count, male_count \
        = calc_researcher_population_pyramid(input_data_name, pub_interval_threshold, base_year)
    pre_female_productivity, pre_male_productivity, pre_female_count, pre_male_count \
        = calc_researcher_population_pyramid(input_data_name, pub_interval_threshold, base_year-1)

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

def save(researcher_population_pyramid, target_past_years, base_year, t_max_projection):

    years = list(sorted(target_past_years)) + list(range(base_year+1, t_max_projection+1))
    cumulative_productivity = set()
    for y in years:
        if y in target_past_years:
            (female_count, male_count) = researcher_population_pyramid[y]
            cumulative_productivity.add(int(min(female_count.keys())))
            cumulative_productivity.add(int(max(female_count.keys())))
            cumulative_productivity.add(int(min(male_count.keys())))
            cumulative_productivity.add(int(max(male_count.keys())))
        elif base_year < y and y <= t_max_projection:
            (future_female_count, future_male_count, female_prob, male_prob, female_newcomer_count, male_newcomer_count) = researcher_population_pyramid[t_max_projection]
            cumulative_productivity.add(int(min(future_female_count[y].keys())))
            cumulative_productivity.add(int(max(future_female_count[y].keys())))
            cumulative_productivity.add(int(min(future_male_count[y].keys())))
            cumulative_productivity.add(int(max(future_male_count[y].keys())))

    min_n, max_n = min(cumulative_productivity), max(cumulative_productivity)

    female_count_lst = []
    male_count_lst = []
    for n in range(min_n, max_n+1):
        f_lst = []
        m_lst = []
        for y in years:
            if y in target_past_years:
                (female_count, male_count) = researcher_population_pyramid[y]
                f_lst.append(female_count.get(n, 0.0))
                m_lst.append(male_count.get(n, 0.0))
            elif base_year < y and y <= t_max_projection:
                (future_female_count, future_male_count, female_prob, male_prob, female_newcomer_count, male_newcomer_count) = researcher_population_pyramid[t_max_projection]
                f_lst.append(future_female_count[y].get(n, 0.0))
                m_lst.append(future_male_count[y].get(n, 0.0))

        female_count_lst.append(f_lst)
        male_count_lst.append(m_lst)

    header = ['Cumulative productivity'] + [str(y) for y in years]
    index = [str(n) for n in range(min_n, max_n+1)]

    with open('./data/female_author_count.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i, row in zip(index, female_count_lst):
            writer.writerow([i] + row)

    with open('./data/male_author_count.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i, row in zip(index, male_count_lst):
            writer.writerow([i] + row)

    (future_female_count, future_male_count, female_prob, male_prob, female_newcomer_count, male_newcomer_count) = researcher_population_pyramid[t_max_projection]

    female_prob_lst = []
    male_prob_lst = []
    for n1 in range(min_n, max_n+1):
        for n2 in range(min_n, max_n+1):
            female_prob_lst.append([n1, n2, female_prob.get(n1, {}).get(n2, 0.0)])
            male_prob_lst.append([n1, n2, male_prob.get(n1, {}).get(n2, 0.0)])

    header = ['Cumulative productivity k1', 'Cumulative productivity k2', 'Transition probability']
    with open('./data/female_trans_prob_from_' + str(base_year-1) + '_to_' + str(base_year) + '.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in female_prob_lst:
            writer.writerow(row)

    with open('./data/male_trans_prob_from_' + str(base_year-1) + '_to_' + str(base_year) + '.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in male_prob_lst:
            writer.writerow(row)

    female_count_lst = []
    male_count_lst = []
    for n in range(min_n, max_n+1):
        female_count_lst.append([n, female_newcomer_count.get(n, 0)])
        male_count_lst.append([n, male_newcomer_count.get(n, 0)])

    header = ['Cumulative productivity', 'Count']
    with open('./data/female_inflow_' + str(base_year) + '.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in female_count_lst:
            writer.writerow(row)

    with open('./data/male_inflow_' + str(base_year) + '.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in male_count_lst:
            writer.writerow(row)

    return

