# README
This document explains how to reproduce the data processing, figures, and tables presented in the following manuscript:

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
# Usage
1. Ensure that the following datasets are placed in the `./data/` directory. These datasets are not publicly available, except where noted.
    - **Orbis dataset**
        - data_sets.pkl
        - This file was extracted from a snapshot of the [Orbis database](https://www.moodys.com/web/en/us/capabilities/company-reference-data/orbis.html) (a subscription-based commercial database). Please contact us to obtain access to this file privately.
    - **World Gender Name Dictionary (WGND)**
        - wgnd_2_0_sources.csv
        - Download from: https://tind.wipo.int/record/49408?ln=en&v=zip
    - **OpenAlex authors by country**
        - author_sample_lst_middle_name_False_last_name_False_special_chars_True.pkl
        - This file was extracted from a snapshot of the [OpenAlex database](https://openalex.org/) (a fully open bibliographic database). Please contact us to obtain access to this file privately.
2. Run all cells in the notebook `gender_inference.ipynb` to train the Complement Naive Bayes classifier for each country.
3. Run all cells in `openalex_gender_assignment.ipynb` to assign binary gender to OpenAlex authors based on the trained classifier.
4. Run all cells in `calc_productive_pyramids.ipynb` to construct researcher population pyramids by country.
5. Run all cells in `make_figs.ipynb` to generate the figures included in the manuscript.
6. Run all cells in `make_tables.ipynb` to generate the tables included in the manuscript.

# Notes
- All computations were performed on a 2019 Mac Pro. Generating the complete set of numerical results took several days.
- Figures of population pyramids for all 58 countries analyzed in our study can be found in the `./figs/` directory.

