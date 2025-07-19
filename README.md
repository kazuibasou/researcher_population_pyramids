# README

This document describes the code used in the following manuscript:

**Kazuki Nakajima and Takayuki Mizuno.** *Researcher Population Pyramids for Tracking Global Demographic and Gender Trajectories*. 2025.

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

Clone this repository:
```
git clone git@github.com:kazuibasou/researcher_population_pyramids.git
```

Go to `researcher_population_pyramids`:
```
cd researcher_population_pyramids
```

# Reproduce our results

0. Run the following command:
```
mkdir data figs
```

1. Ensure that the following datasets are placed in the `researcher_population_pyramids/data/` directory. Due to the large size of the raw OpenAlex data and licensing restrictions for the Orbis data, the full dataset cannot be publicly shared (therefore, the `data` directory is empty here). However, a curated version of the data sufficient to reproduce the results, figures, and tables presented in our manuscript can be made available privately upon reasonable request to the authors, provided that applicable data use agreements and licensing terms are met.
    - **Orbis data**
        - data_sets.pkl
            - This file was extracted from a snapshot of the [Orbis database](https://www.moodys.com/web/en/us/capabilities/company-reference-data/orbis.html) (a subscription-based commercial database). 
    - **World Gender Name Dictionary (WGND)**
        - wgnd_2_0_sources.csv
            - Download from: https://tind.wipo.int/record/49408?ln=en&v=zip
    - **OpenAlex data**
        - openalex_affiliation_to_country.pickle
        - openalex_author_name.pickle
        - openalex_work_data.pickle
            - These files were extracted from its September, 2024 snapshot of the [OpenAlex database](https://openalex.org/) (a fully open bibliographic database).

2. Run all cells in the notebook `gender_inference.ipynb` sequentially to train the Complement Naive Bayes classifier for each country.

3. Run all cells in `openalex_gender_assignment.ipynb` sequentially to assign binary gender to OpenAlex authors based on the trained classifier.

4. Run all cells in `calc_productive_pyramids.ipynb` sequentially to construct researcher population pyramids by country.

5. Run all cells in `make_figs.ipynb` sequentially to generate the figures included in the manuscript.

6. Run all cells in `make_tables.ipynb` sequentially to generate the tables included in the manuscript.

# Notes
- All computations were performed on a 2019 Mac Pro. Generating the complete set of numerical results took several days.
- Figures of population pyramids in 2023 for all 58 countries analyzed in our study can be found at the `./figs/` directory.

