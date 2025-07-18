import pickle
import re
from unidecode import unidecode
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
import numpy as np
import random
import pycountry
import json
import openpyxl
from datetime import datetime
import csv
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split
from ftfy import fix_text

data_dir = "./data/"

def name_cleaning(name):
    name = fix_text(name)  # Unicode修復
    name = unidecode(name)  # アクセント除去
    name = re.sub(r"[^a-zA-Z]", "", name)  # 英字以外除去

    return name.title()  # 頭文字だけ大文字に

def split_name_into_words(name):

    return [str(word) for word in list(name.split(' ')) if len(word) > 0]

def tokenize_name(first_name, middle_name, last_name, clearning_special_chars):
    #  Example: (kazuki)[daniel]<nakajima>

    if clearning_special_chars:
        first_name_ = name_cleaning(first_name)
        middle_name_ = name_cleaning(middle_name)
        last_name_ = name_cleaning(last_name)
    else:
        first_name_ = first_name
        middle_name_ = middle_name
        last_name_ = last_name

    if middle_name == "" and last_name == "":
        return "(" + first_name_ + ")"
    else:
        return "(" + first_name_ + ")" + "[" + middle_name_ + "]" + "<" + last_name_ + ">"

def construct_data_sets(make_pickle=False):

    if make_pickle:
        df = pd.read_csv("./data/training_data_with_country_of_nationality.csv", sep='\t', low_memory=False)

        print("training_data_with_country_of_nationality.csv", df.shape)

        with open("./data/training_data_with_country_of_nationality.pkl", "wb") as f:
            pickle.dump(df, f)

        df = pd.read_csv("./data/training_data_with_country.csv", sep='\t', low_memory=False)

        print("training_data_with_country.csv", df.shape)

        with open("./data/training_data_with_country.pkl", "wb") as f:
            pickle.dump(df, f)

    with open("./data/training_data_with_country_of_nationality.pkl", 'rb') as f:
        df = pickle.load(f)

    df = df[(df["salutation"] == "Mr") | (df["salutation"] == "Ms")]
    df["salutation"] = df["salutation"].replace("Ms", 0)
    df["salutation"] = df["salutation"].replace("Mr", 1)
    df = df.drop(
        ["suffix", "full_name", "gender", "age", "age_bracket"],
        axis=1)

    d = df.to_dict('index')
    alpha_2_country_code = read_alpha_2_country_code_mapping()

    countries = set([])
    for i in d:
        c_lst = str(d[i]["country_of_nationality"]).split(";")
        countries = countries | set(c_lst)
        c_lst = str(d[i]["country"]).split(";")
        countries = countries | set(c_lst)

    data_by_country = {}

    for c_ in countries:
        if c_ == 'None' or c_ == 'nan' or c_ == 'N/A':
            continue
        c = alpha_2_country_code[c_]
        if c in {'None', 'nan', 'N/A'}:
            continue
        data_by_country[c] = [[], []]

    female_male_count = {}
    for i in d:
        #c_set = set(list(str(d[i]["country_of_nationality"]).split(";")) + list(str(d[i]["country"]).split(";")))
        c_set = set(list(str(d[i]["country_of_nationality"]).split(";")))
        if 'None' in c_set:
            c_set.remove('None')
        if 'nan' in c_set:
            c_set.remove('nan')
        if 'N/A' in c_set:
            c_set.remove('N/A')

        if len(c_set) != 1:
            continue

        country = list(c_set)[0]
        country_code = alpha_2_country_code[country]

        if country_code in {'None', 'nan', 'N/A'}:
            continue

        gender = int(d[i]["salutation"])
        first_name = str(d[i]["first_name"])
        middle_name = str(d[i]["middle_name"])
        last_name = str(d[i]["last_name"])

        if first_name == 'nan':
            continue

        try:
            year_of_birth = str(int(datetime.strptime(str(d[i]["date_of_birth"]), "%Y-%m-%d %H:%M:%S").year))
        except:
            year_of_birth = ""

        data = [
            gender,
            first_name,
            middle_name,
            last_name,
            year_of_birth,
        ]

        data_by_country[country_code][gender].append(data)

        if country not in female_male_count:
            female_male_count[country] = [0, 0]
        female_male_count[country][int(d[i]["salutation"])] += 1

    n = sum([len(data_by_country[c][0]) + len(data_by_country[c][1]) for c in data_by_country])

    print(n, flush=True)

    with open("./data/data_sets.pkl", "wb") as f:
        pickle.dump(data_by_country, f)

    return

def construct_train_validation_test_sets_across_countries(
        data_by_country, country_lst, test_size, validation_size,
        random_state, shuffle,
        include_middle_name, include_last_name, clearning_special_chars,
        min_ratio_minority_to_majority,
        arab_league, arab_minority_group
        ):

    # Helper function to extract and tokenize data (returns string list for CountVectorizer)
    def _extract_and_tokenize_gender_split(data_list, gender_label, country):
        corpus = []
        labels = []
        name_country_list = []
        for item in data_list:

            first_name_data = item[1]
            middle_name_data = item[2] if len(item) > 2 else ""
            last_name_data = item[3] if len(item) > 3 else ""

            current_first_name = item[1]
            current_middle_name = item[2]
            current_last_name = item[3]

            if not include_middle_name:
                current_middle_name = ""
            if not include_last_name:
                current_last_name = ""

            name_str = tokenize_name(current_first_name, current_middle_name, current_last_name,
                                     clearning_special_chars)
            corpus.append(name_str)
            labels.append(gender_label)
            name_country_list.append((current_first_name, country))
        return corpus, labels, name_country_list

    arab_countries_set = set(arab_league) - set(arab_minority_group)

    # --- 1. Prepare Data for Each Country (Raw splits) ---
    # Store raw data for each country after splitting
    country_raw_splits = {}

    # Store all training data (names as strings) for non-minority Arab countries
    all_arab_train_corpus_raw = []
    all_arab_train_labels_raw = []

    rng = np.random.default_rng(random_state)  # Use numpy for better randomness control

    for country in country_lst:

        # Get raw data for female (0) and male (1)
        # Note: data_by_country[country][0] and [1] are lists of lists like [[gender, fname, mname, lname, yob], ...]
        female_raw_data = list(data_by_country[country][0])
        male_raw_data = list(data_by_country[country][1])

        if shuffle:
            rng.shuffle(female_raw_data)
            rng.shuffle(male_raw_data)

        # --- Check for sufficient data ---
        if len(female_raw_data) < (test_size + validation_size) or \
                len(male_raw_data) < (test_size + validation_size):
            print(
                f"WARNING: Not enough data for country {country} to form test and validation sets of size {test_size} each. Skipping this country.")
            country_raw_splits[country] = None  # Mark as skipped
            continue

        # --- Data Splitting using sklearn's train_test_split ---
        # 1. Split off test set
        f_remaining, f_test = train_test_split(female_raw_data, test_size=test_size, random_state=random_state)
        m_remaining, m_test = train_test_split(male_raw_data, test_size=test_size, random_state=random_state)

        # 2. Split remaining into validation and training
        f_train, f_val = train_test_split(f_remaining, test_size=validation_size, random_state=random_state)
        m_train, m_val = train_test_split(m_remaining, test_size=validation_size, random_state=random_state)

        # --- Imbalance handling for individual country's training set (non-Arab minority) ---
        if country not in arab_minority_group:
            n_f_train = len(f_train)
            n_m_train = len(m_train)

            if n_f_train > 0 and n_m_train > 0:
                # Downsample majority if ratio is too low
                if n_f_train < n_m_train:  # Female is minority
                    if float(n_f_train) / n_m_train < min_ratio_minority_to_majority:
                        target_m_size = int(n_f_train / min_ratio_minority_to_majority)
                        m_train = rng.choice(m_train, size=min(target_m_size, n_m_train), replace=False).tolist()
                else:  # Male is minority or equal
                    if float(n_m_train) / n_f_train < min_ratio_minority_to_majority:
                        target_f_size = int(n_m_train / min_ratio_minority_to_majority)
                        f_train = rng.choice(f_train, size=min(target_f_size, n_f_train), replace=False).tolist()
            elif n_f_train == 0 or n_m_train == 0:
                print(
                    f"WARNING: No effective training data for one gender after split for {country}. Imbalance handling skipped for this country's individual train set.")

        # Store split raw data
        country_raw_splits[country] = {
            'train': {'female': f_train, 'male': m_train},
            'val': {'female': f_val, 'male': m_val},
            'test': {'female': f_test, 'male': m_test}
        }

        # Accumulate raw training data for non-minority Arab countries (for pooled Arab training set)
        if country in arab_countries_set:
            all_arab_train_corpus_raw.extend(_extract_and_tokenize_gender_split(f_train, 0, country)[0])
            all_arab_train_labels_raw.extend(_extract_and_tokenize_gender_split(f_train, 0, country)[1])
            all_arab_train_corpus_raw.extend(_extract_and_tokenize_gender_split(m_train, 1, country)[0])
            all_arab_train_labels_raw.extend(_extract_and_tokenize_gender_split(m_train, 1, country)[1])

    # --- 2. Build Global/Pooled Feature Transformers ---
    global_fit_corpus_raw = []  # Raw strings for vectorizer fit
    global_fit_labels_raw = []  # Keep labels for potential full global imbalance handling if needed later

    # Add the (potentially downsampled) pooled Arab training corpus
    if all_arab_train_corpus_raw:  # Ensure pooled Arab data exists before adding
        global_fit_corpus_raw.extend(all_arab_train_corpus_raw)
        global_fit_labels_raw.extend(all_arab_train_labels_raw)

    # Collect training data from ALL other countries found in data_by_country
    # (Excluding the ones already pooled as Arab, and excluding any data that went to test/validation sets)
    # We should iterate through all countries available in data_by_country keys

    # Create a set of countries whose training data is already part of all_arab_train_corpus_raw
    # This set will be used to avoid duplicating data.
    arab_countries_in_pooled = arab_countries_set  # Assuming arab_countries_set already accumulated its raw train data

    for country in data_by_country.keys():  # Iterate through ALL countries available in the input data_by_country
        # Skip if the country was marked as skipped during initial raw split (due to insufficient data for T/V sets)
        if country_raw_splits.get(country) is None:
            continue

        # Skip if this country's training data has already been added to the pooled Arab training corpus
        if country in arab_countries_in_pooled:
            continue

        # For arab_minority_group countries, their training data is the *pooled* arab_train_corpus,
        # so we don't add their individual training data here again.
        # This means, for these countries, their X_train will use global_count_vect.transform(all_arab_train_corpus_raw).
        # We need to make sure their individual data is not added to global_fit_corpus_raw for fitting.
        if country in arab_minority_group:
            continue

        # For all other countries, add their individual training data (which already had imbalance handling)
        # Note: country_raw_splits[country]['train'] already holds the imbalance-handled portion
        f_train_data = country_raw_splits[country]['train']['female']
        m_train_data = country_raw_splits[country]['train']['male']

        global_fit_corpus_raw.extend(_extract_and_tokenize_gender_split(f_train_data, 0, country)[0])
        global_fit_labels_raw.extend(_extract_and_tokenize_gender_split(f_train_data, 0, country)[1])
        global_fit_corpus_raw.extend(_extract_and_tokenize_gender_split(m_train_data, 1, country)[0])
        global_fit_labels_raw.extend(_extract_and_tokenize_gender_split(m_train_data, 1, country)[1])

    # --- Global imbalance handling (optional, but good for very imbalanced global corpus) ---
    # You might want to add global imbalance handling here if the total global_fit_corpus_raw
    # becomes highly imbalanced. This would be a final downsampling of the global majority.
    # For now, it's not explicitly in your existing logic, but consider it if needed.

    if not global_fit_corpus_raw:
        print("ERROR: No global training corpus could be formed for feature transformation. Exiting.")
        return {}

    # Calculate n_max for global CountVectorizer (based on character length of names)
    n_max = 0
    for name_str in global_fit_corpus_raw:
        n_max = int(max(len(name_str), n_max))
    n_gram_range = (1, n_max)

    # Fit global CountVectorizer and TfidfTransformer once
    global global_count_vect, global_tfidf_transformer
    global_count_vect = CountVectorizer(analyzer='char', ngram_range=n_gram_range)
    X_global_train_counts = global_count_vect.fit_transform(global_fit_corpus_raw)  # Fit on raw strings
    global_tfidf_transformer = TfidfTransformer()
    global_tfidf_transformer.fit(X_global_train_counts)

    # --- 3. Process Each Country's Data for Model Training/Evaluation ---
    all_country_datasets = {}  # To store results for all countries

    for country in country_lst:
        if country_raw_splits.get(country) is None:  # Skip if country was skipped earlier
            all_country_datasets[country] = None
            continue

        # Get the pre-split raw data for current country
        splits = country_raw_splits[country]

        # Tokenize and extract corpus and labels for current country's splits
        train_corpus_raw, y_train_list, _ = _extract_and_tokenize_gender_split(splits['train']['female'], 0, country)
        temp_corpus_raw, temp_labels, _ = _extract_and_tokenize_gender_split(splits['train']['male'], 1, country)
        train_corpus_raw.extend(temp_corpus_raw)
        y_train_list.extend(temp_labels)

        val_corpus_raw, y_val_list, val_name_country_lst = _extract_and_tokenize_gender_split(splits['val']['female'],
                                                                                              0, country)
        temp_corpus_raw, temp_labels, temp_name_country_lst = _extract_and_tokenize_gender_split(splits['val']['male'],
                                                                                                 1, country)
        val_corpus_raw.extend(temp_corpus_raw)
        y_val_list.extend(temp_labels)
        val_name_country_lst.extend(temp_name_country_lst)

        test_corpus_raw, y_test_list, test_name_country_lst = _extract_and_tokenize_gender_split(
            splits['test']['female'], 0, country)
        temp_corpus_raw, temp_labels, temp_name_country_lst = _extract_and_tokenize_gender_split(splits['test']['male'],
                                                                                                 1, country)
        test_corpus_raw.extend(temp_corpus_raw)
        y_test_list.extend(temp_labels)
        test_name_country_lst.extend(temp_name_country_lst)

        # --- Apply Feature Transformation using GLOBAL transformers ---
        if country in arab_minority_group:
            # For arab_minority_group, X_train uses the globally processed pooled Arab training data
            X_train_transformed_counts = global_count_vect.transform(all_arab_train_corpus_raw)
            X_train = global_tfidf_transformer.transform(X_train_transformed_counts)
            y_train = np.array(all_arab_train_labels_raw, dtype=int)
            print(
                f"DEBUG: Using pooled Arab training set (size F:{np.sum(y_train == 0)}, M:{np.sum(y_train == 1)}) for {country}.")
        else:
            # For other countries, X_train uses its own country's training data
            X_train_transformed_counts = global_count_vect.transform(train_corpus_raw)
            X_train = global_tfidf_transformer.transform(X_train_transformed_counts)
            y_train = np.array(y_train_list, dtype=int)

        X_val_transformed_counts = global_count_vect.transform(val_corpus_raw)
        X_val = global_tfidf_transformer.transform(X_val_transformed_counts)

        X_test_transformed_counts = global_count_vect.transform(test_corpus_raw)
        X_test = global_tfidf_transformer.transform(X_test_transformed_counts)

        y_val = np.array(y_val_list, dtype=int)
        y_test = np.array(y_test_list, dtype=int)

        # --- Verification of sizes ---
        f_test_extracted_size = np.sum(y_test == 0)
        m_test_extracted_size = np.sum(y_test == 1)
        f_val_extracted_size = np.sum(y_val == 0)
        m_val_extracted_size = np.sum(y_val == 1)
        f_train_extracted_size = np.sum(y_train == 0)
        m_train_extracted_size = np.sum(y_train == 1)
        ratio_minority_to_majority = (float(min(f_train_extracted_size, m_train_extracted_size)) /
                                      max(f_train_extracted_size, m_train_extracted_size))

        if (f_test_extracted_size != test_size or m_test_extracted_size != test_size
                or f_val_extracted_size != test_size or m_val_extracted_size != test_size
            or ratio_minority_to_majority < min_ratio_minority_to_majority):
            print(f"ERROR: {country}: Female train/val/test sizes: F:{f_train_extracted_size}/{f_val_extracted_size}/{f_test_extracted_size} | Male train/val/test sizes: M:{m_train_extracted_size}/{m_val_extracted_size}/{m_test_extracted_size} | Ratio of minority train size to majority train size: {ratio_minority_to_majority}")
            exit()

        print(
            f"{country}: Female train/val/test sizes: F:{f_train_extracted_size}/{f_val_extracted_size}/{f_test_extracted_size} | Male train/val/test sizes: M:{m_train_extracted_size}/{m_val_extracted_size}/{m_test_extracted_size} | Ratio of minority train size to majority train size: {ratio_minority_to_majority}")

        # Strict check for exact test_size and validation_size
        if f_test_extracted_size != test_size or m_test_extracted_size != test_size:
            print(
                f"ERROR: Test size for {country} is incorrect. Expected {test_size}, got F:{f_test_extracted_size}, M:{m_test_extracted_size}. This means not enough data was available to extract exact sizes.")
            all_country_datasets[country] = None
            continue
        if f_val_extracted_size != validation_size or m_val_extracted_size != validation_size:
            print(
                f"ERROR: Validation size for {country} is incorrect. Expected {validation_size}, got F:{f_val_extracted_size}, M:{m_val_extracted_size}. This means not enough data was available to extract exact sizes.")
            all_country_datasets[country] = None
            continue

        # Save data for debugging or later use
        f_name = ("./data/" + str(country).lower() + "_train_val_test_sets"
                  + "_middle_name_" + str(include_middle_name)
                  + "_last_name_" + str(include_last_name)
                  + "_special_chars_" + str(clearning_special_chars)
                  + ".pkl")

        with open(f_name, "wb") as f:
            pickle.dump((X_train, y_train, X_val, y_val, X_test, y_test,
                         test_name_country_lst, val_name_country_lst), f)

        all_country_datasets[country] = {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test,
            'test_name_country_lst': test_name_country_lst,
            'val_name_country_lst': val_name_country_lst
        }

    # Save the global vectorizers/transformers separately for later use (e.g., OpenAlex transformation)
    with open("./data/global_feature_transformers.pkl", "wb") as f:
        pickle.dump((global_count_vect, global_tfidf_transformer), f)

    return

def construct_alpha_2_country_code_mapping():
    country_code = {}
    for country in pycountry.countries:
        country_code[country.name] = country.alpha_2

    manually_filled_country_code = {
        "Vietnam": "VN",
        "Czech Republic": "CZ",
        "Kosovo": "N/A",
        "Turkey": "TR",
        "Hong Kong SAR, China": "HK",
        "Moldova Republic of": "MD",
        "Korea Republic of": "KR",
        "Iran (Islamic Republic of)": "IR",
        "Taiwan, China": "TW",
        "Libyan Arab Jamahiriya": "LY",
        "Venezuela": "VE",
        "Tanzania United Republic of": "TZ",
        "Bolivia": "BO",
        "Palestinian Territory": "PS",
        "Congo Democratic Republic of": "CD",
        "Virgin Islands (British)": "VG",
        "Swaziland": "SZ",
        "Cape Verde": "CV",
        "Korea Democratic People's Republic of": "KP",
        "Macao SAR, China": "MO",
        "Turks-and-Caicos": "TC",
        "East Timor": "TL",
        "Vatican": "VA",
        "Micronesia (Federated States of)": "FM",
        "United States of America": "US",
        "Islamic Republic of Iran": "IR",
        "Federated States of Micronesia": "FM",
        "Virgin Islands (U.S.)": "VI",
        "Palestinian Territories": "PS",
        "European Community": "N/A",
        "Republic of Korea": "KR",
        "Democratic People'S Republic of Korea": "KP",
        "United Republic of Tanzania": "TZ",
        "Lao People'S Democratic Republic": "LA",
        "Falkland Islands": "FK",
        "Vatican City State/Holy See": "VA",
        "Bonaire, Saba and Saint Eustatius": "BQ",
        "Saint Vincent and The Grenadines": "VC",
        "Republic of Moldova": "MD",
        "Supranational": "N/A",
        "Us Minor Outlying Islands": "UM",
        "Guinea Bissau": "GW",
        "Democratic Republic of Congo": "CD",
        "Island of Man": "IM",
    }

    alpha_2_country_code_mapping = {}
    for country in country_code:
        alpha_2_country_code_mapping[country] = country_code[country]

    for country in manually_filled_country_code:
        alpha_2_country_code_mapping[country] = manually_filled_country_code[country]

    with open('./data/country_alpha_2_code.json', mode='w', encoding='utf-8') as f:
        json.dump(alpha_2_country_code_mapping, f)

    return alpha_2_country_code_mapping

def read_alpha_2_country_code_mapping():
    with open('./data/country_alpha_2_code.json', mode='r', encoding='utf-8') as f:
        alpha_2_country_code_mapping = json.load(f)

    return alpha_2_country_code_mapping

def construct_wgnd_val_test_data(country_lst, test_size, clearning_special_chars, random_state, min_name_samples):

    name_gender_weight_by_country = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))

    csv_path = './data/wgnd_2_0_sources.csv'
    with open(csv_path, newline='', encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            first_name = name_cleaning(str(row[0]))
            country = str(row[1])
            gender = str(row[2])
            nobs = float(row[4])

            if len(first_name) <= 1 or gender not in {'F', 'M'}:
                continue

            if gender == 'F':
                gender = 0
            else:
                gender = 1

            name_gender_weight_by_country[country][gender][first_name] += nobs

    print("Read wgnd name-and-gender data.")

    val_data, test_data = [], []
    n_c = 0
    for country in country_lst:

        female_name_lst = list(name_gender_weight_by_country[country][0].keys())
        male_name_lst = list(name_gender_weight_by_country[country][1].keys())

        if len(female_name_lst) < min_name_samples or len(male_name_lst) < min_name_samples:
            continue

        female_name_weights = [name_gender_weight_by_country[country][0][first_name] for first_name in female_name_lst]
        male_name_weights = [name_gender_weight_by_country[country][1][first_name] for first_name in male_name_lst]

        country_val_rng = random.Random(random_state + hash(country) % (2 ** 30))  # Validation-specific RNG
        country_test_rng = random.Random(random_state + hash(country + "_test") % (2 ** 30))  # Test-specific RNG

        female_val_samples = country_val_rng.choices(female_name_lst, k=test_size, weights=female_name_weights)
        male_val_samples = country_val_rng.choices(male_name_lst, k=test_size, weights=male_name_weights)

        for k in range(0, test_size):
            val_data.append((female_val_samples[k], country, 0))
            val_data.append((male_val_samples[k], country, 1))

        female_test_samples = country_test_rng.choices(female_name_lst, k=test_size, weights=female_name_weights)
        male_test_samples = country_test_rng.choices(male_name_lst, k=test_size, weights=male_name_weights)

        for k in range(0, test_size):
            test_data.append((female_test_samples[k], country, 0))
            test_data.append((male_test_samples[k], country, 1))

        if (len([x for x in test_data if x[1] == country and x[2] == 0]) != test_size
            or len([x for x in test_data if x[1] == country and x[2] == 1]) != test_size
            or len([x for x in val_data if x[1] == country and x[2] == 0]) != test_size
            or len([x for x in val_data if x[1] == country and x[2] == 1]) != test_size):
            print("ERROR: test size is unvalid.")
            print(len([x for x in test_data if x[1] == country and x[2] == 0]),
                  len([x for x in test_data if x[1] == country and x[2] == 1]),
                  len([x for x in val_data if x[1] == country and x[2] == 0]),
                  len([x for x in val_data if x[1] == country and x[2] == 1]))
            return

        # print(country,
        #       len([x for x in test_data if x[1] == country and x[2] == 0]),
        #       len([x for x in test_data if x[1] == country and x[2] == 1]),
        #       len([x for x in val_data if x[1] == country and x[2] == 0]),
        #       len([x for x in val_data if x[1] == country and x[2] == 1])
        #       )

        n_c += 1

    print("Number of countries:", n_c)

    df = pd.DataFrame(test_data, columns=['First name', 'Country', 'Gender'])
    df.to_excel('./data/wgnd_test_data_for_api.xlsx', index=False)

    df = pd.DataFrame(val_data, columns=['First name', 'Country', 'Gender'])
    df.to_excel('./data/wgnd_val_data_for_api.xlsx', index=False)

    wgnd_val_name_corpus = defaultdict(list)
    wgnd_y_val = defaultdict(list)

    for (first_name, c_code, gender) in val_data:

        if len(first_name) <= 1:
            continue

        middle_name = ""
        last_name = ""
        name_token = tokenize_name(first_name, middle_name, last_name,
                                   clearning_special_chars=clearning_special_chars)

        wgnd_val_name_corpus[c_code].append(name_token)
        wgnd_y_val[c_code].append(gender)

    f_name = ("./data/wgnd_val_data.pkl")

    with open(f_name, "wb") as f:
        pickle.dump((wgnd_val_name_corpus, wgnd_y_val), f)

    wgnd_test_name_corpus = defaultdict(list)
    wgnd_y_test = defaultdict(list)

    for (first_name, c_code, gender) in test_data:

        if len(first_name) <= 1:
            continue

        middle_name = ""
        last_name = ""
        name_token = tokenize_name(first_name, middle_name, last_name,
                                   clearning_special_chars=clearning_special_chars)

        wgnd_test_name_corpus[c_code].append(name_token)
        wgnd_y_test[c_code].append(gender)

    f_name = ("./data/wgnd_test_data.pkl")

    with open(f_name, "wb") as f:
        pickle.dump((wgnd_test_name_corpus, wgnd_y_test), f)

    return

def read_pickle_file(f_name):

    f_path = data_dir + f_name
    with open(f_path, mode="rb") as f:
        d = pickle.load(f)

    return d