# Real Estate Market Analysis and Value Evaluation

## Project Overview

This project is a Python-based exploratory data analysis project using a real estate dataset. The aim is to clean and analyse housing data to understand price distribution, city-level price differences, property characteristics, and possible value-related patterns.

This project was completed as part of my professional learning portfolio. It demonstrates practical data cleaning, feature engineering, data analysis, and visualisation skills.

## Dataset

The dataset contains house sales listing information, including:

- city
- district
- address
- area
- total price
- unit price
- room type
- floor information
- building year
- house orientation
- listing URL

Some raw fields contain text and numbers together. Therefore, the project includes feature extraction to convert important values into usable numeric fields.

## Key Analysis Questions

This project investigates the following questions:

1. What is the general distribution of house prices?
2. Which numerical variables are more related to total price and unit price?
3. Which cities have higher median unit prices?
4. What are the characteristics of different price segments?
5. Which room types are most common?
6. How does home orientation relate to unit price?
7. How does building age relate to unit price?
8. Do municipality-level cities show higher prices than non-municipality cities?

## Data Cleaning and Feature Engineering

The script performs the following data cleaning steps:

- removes rows with missing essential values
- removes duplicated listing URLs when available
- extracts numeric area, total price, and unit price values
- extracts bedroom and living room counts from room type text
- extracts building year and estimates building age
- creates building age groups
- creates price segments
- creates simplified city labels and city type groups
- filters unrealistic outliers

## Output Files

When the script is run, the following output files are created in the `outputs_real_estate_project` folder:

- `01_raw_dataset_overview.csv`
- `02_cleaning_summary.csv`
- `03_cleaned_house_sales.csv`
- `04_numeric_summary.csv`
- `05_correlation_matrix.csv`
- `06_city_price_summary.csv`
- `07_price_segment_summary.csv`
- `08_room_type_counts.csv`
- `09_room_type_price_summary.csv`
- `10_orientation_price_summary.csv`
- `11_building_age_price_summary.csv`
- `12_city_type_summary.csv`
- `13_project_summary.txt`

The script also creates several chart images, including price distribution, correlation matrix, top city prices, room type comparison, orientation comparison, building age comparison, and municipality versus non-municipality comparison.

## How to Run

Place `house_sales.csv` in the same folder as the Python script, then run:

```bash
python real_estate_market_analysis_en.py
```

## Professional Learning Reflection

This project helped me practise a realistic data analysis workflow. I learned how to clean raw data, create new features, handle missing values, compare groups, and present findings visually. I also improved my ability to structure a project in a way that can be shown in a professional portfolio.
