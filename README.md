<h1 align="center">
<i>Researcher Population Pyramids for Tracking Global Demographic and Gender Trajectories</i>
</h1>

<!-- <p align="center">
<a href="https://github.com/kazuibasou/hyper-dk-series/blob/main/LICENSE" target="_blank">
<img alt="License: MIT" src="https://img.shields.io/github/license/kazuibasou/hyperneo">
</a>

<a href="https://arxiv.org/abs/2106.12162" target="_blank">
<img alt="ARXIV: 2106.12162" src="https://img.shields.io/badge/arXiv-2106.12162-red.svg">
</a>

</p> -->

The researcher population pyramids framework is a visualization and diagnostic tool for tracking global demographic and gender trajectories using publication data. 
This framework provides a timely snapshot of the present state of demographics and gender balance of the global academic ecosystem and simulates its potential trajectories.

If you use this code, please cite

- Kazuki Nakajima and Takayuki Mizuno. Researcher Population Pyramids for Tracking Global Demographic and Gender Trajectories. *arXiv* (2025).


# Requirements

The code has been tested and confirmed to work in the following environment:

## Python and Required Libraries

- Python 3.10.15
- ftfy==6.3.1
- matplotlib==3.10.3
- nomquamgender==0.1.4
- numpy==2.3.1
- openpyxl==3.1.5
- pandas==2.3.1
- pycountry==24.6.1
- scikit_learn==1.7.0
- Unidecode==1.3.8

## Operating System

- macOS 14.4


# Build

1. Clone this repository:
```
git clone git@github.com:kazuibasou/researcher_population_pyramids.git
```

2. Go to `researcher_population_pyramids`:
```
cd researcher_population_pyramids
```

3. Run the following commands sequentially:
```
mkdir -p ./framework/figs
mkdir -p ./reproduction/data
mkdir -p ./reproduction/figs
```

This generates the following structure of the directory.

    researcher_population_pyramids/
    ├ framework/
       ├ data/
       └ figs/
    └  reproduction/
       ├ data/
       └ figs/

The `framework` directory provides a set of code to construct and visualize researcher population pyramids for various countries, based on author and publication data extracted from some bibliographic database (e.g., [OpenAlex](https://openalex.org/)). 
It is designed for general application and further development of the framework.

The `reproduction` directory contains the code necessary to reproduce the results, figures, and tables presented in our manuscript. 
Please note that full reproduction requires access to the proprietary, restricted-access data used in our study.


# Usage of the framework

## Input file

To utilize the framework, you need to provide a sample of authors for a single country in a specific JSON Lines (JSONL) format. 
The main input file, which should be named `author_sample_lst.jsonl`, must be placed in the `researcher_population_pyramids/framework/data/` directory.

Each line in this file corresponds to an author's data. 
The format adheres to the following structure:

```jsonl
{"id": "author_id_1", "gender": 0, "pub_date_lst": ["yyyy-mm-dd", "yyyy-mm-dd", ...]}
{"id": "author_id_2", "gender": 1, "pub_date_lst": ["yyyy-mm-dd", "yyyy-mm-dd", ...]}
...
```

Field Descriptions:

- ``id``: A unique identifier for the author. This can be any string type (no other requirements).
- ``gender``: The inferred gender label for the author (integer type).
    - ``0`` indicates female.
    - ``1`` indicates male.
    - Note: As mentioned in the manuscript, our framework is based on a binary conception of gender and does not capture the full spectrum of gender identities.
- ``pub_date_lst``: A list of the publication dates of the author's papers. Each publication date must be a string in the format ``yyyy-mm-dd``.

For a complete example of this format, please refer to the ``author_sample_lst.jsonl`` file located within the ``researcher_population_pyramids/framework/data/`` directory of this repository.

For author and publication data, for example, you may use [OpenAlex](https://openalex.org/). 
To infer the gender of an author, for example, you may use an open-source classifier called [CCT classifier](https://github.com/ianvanbuskirk/nomquamgender) that estimates the gender of a first name. 
(Note: While our manuscript primarily used a custom Complement Naive Bayes classifier, the CCT classifier offers an open-source alternative for gender inference.) 
We provide the tuned threshold values for the CCT classifier based on benchmark data from 61 countries, in the `researcher_population_pyramids/framework/data/` directory of this repository (see the file `cct_prob_threshold.json`). 
For more details, please refer to Section S1 of the supplementary information in our manuscript.

## Constructing researcher population pyramids

Please see the notebook [construct_population_pyramids.ipynb](https://github.com/kazuibasou/researcher_population_pyramids/blob/main/framework/construct_population_pyramids.ipynb) in the folder ``researcher_population_pyramids/framework/`` for instructions on using the framework in Python.

# Reproduce our results

1. Run the following command:
```
cd researcher_population_pyramids/reproduction
```

2. Ensure that the following datasets are placed in the `researcher_population_pyramids/reproduction/data/` directory. Due to the large size of the raw OpenAlex data and licensing restrictions for the Orbis data, the full dataset cannot be publicly shared (therefore, the `researcher_population_pyramids/reproduction/data` directory is empty here).
    - **Orbis data**
        - data_sets.pkl
            - This file was extracted from its April, 2024 snapshot of the [Orbis database](https://www.moodys.com/web/en/us/capabilities/company-reference-data/orbis.html) (a subscription-based commercial database). 
    - **World Gender Name Dictionary (WGND)**
        - wgnd_2_0_sources.csv
            - Download from: https://tind.wipo.int/record/49408?ln=en&v=zip
    - **OpenAlex data**
        - openalex_affiliation_to_country.pickle
        - openalex_author_name.pickle
        - openalex_work_data.pickle
            - These files were extracted from its September, 2024 snapshot of the [OpenAlex database](https://openalex.org/) (a fully open bibliographic database).

3. Run all cells in the notebook `researcher_population_pyramids/reproduction/gender_inference.ipynb` sequentially to train the Complement Naive Bayes classifier for each country.

4. Run all cells in `researcher_population_pyramids/reproduction/openalex_gender_assignment.ipynb` sequentially to assign binary gender to OpenAlex authors based on the trained classifier.

5. Run all cells in `researcher_population_pyramids/reproduction/construct_population_pyramids.ipynb` sequentially to construct researcher population pyramids by country.

6. Run all cells in `researcher_population_pyramids/reproduction/make_figs.ipynb` sequentially to generate the figures included in the manuscript.

7. Run all cells in `researcher_population_pyramids/reproduction/make_tables.ipynb` sequentially to generate the tables included in the manuscript.

## Notes
- All computations were performed on a 2019 Mac Pro. Generating the complete set of numerical results took several days.
- Figures of population pyramids for all 58 countries analyzed in our study will be generated within the `researcher_population_pyramids/reproduction/figs/` directory, where a dedicated subdirectory (named after the country's alpha-two code) will be created for each country's figures.
