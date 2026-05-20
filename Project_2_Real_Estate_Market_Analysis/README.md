# Real Estate Market Analysis and Value Evaluation

## Project Overview

This project is a Python-based exploratory data analysis project using a real estate dataset. The aim is to clean and analyse housing data to understand price distribution, city-level price differences, property characteristics, and possible value-related patterns.

This project was completed as part of my professional learning portfolio. It demonstrates practical data cleaning, feature engineering, data analysis, and visualisation skills.

## Dataset

The dataset contains house sales listing information, including:

- city
- province
- address
- area
- floor
- rooms
- orientation
- building year
- total price
- unit price
- listing URL

Some fields in the original dataset are text-based, so the project extracts numerical values from area, price, unit price, room type, floor, and building year fields.

## Tools Used

- Python
- pandas
- matplotlib

## Workflow

The project followed these steps:

1. Load the raw house sales dataset.
2. Check the dataset structure and missing values.
3. Extract numerical values from text-based fields.
4. Create new analytical features.
5. Remove duplicated listings and unrealistic values.
6. Generate descriptive statistics.
7. Create grouped summary tables.
8. Produce visual charts.
9. Save cleaned data, summary tables, and project evidence.

## Feature Engineering

The project created several new features, including:

- `area_m2`
- `total_price_10k_yuan`
- `unit_price_yuan_m2`
- `bedrooms`
- `living_rooms`
- `building_age`
- `building_age_group`
- `room_type`
- `orientation_group`
- `city_type`
- `price_segment`

These features made the dataset more useful for analysis and comparison.

## Analysis Questions

The project explored several questions:

- What is the overall distribution of house prices?
- Which variables are most related to total house price?
- Which cities have higher median unit prices?
- Which room types are most common?
- How do price segments differ in area and building age?
- Does home orientation show different median unit prices?
- How does building age relate to unit price?
- Are municipality listings different from non-municipality listings?

## Output Files

The project generated several evidence files, including:

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
- `a1_correlation_matrix.png`
- `a2_total_price_distribution.png`
- `a2_unit_price_boxplot.png`
- `a3_top_city_unit_prices.png`
- `a4_median_area_by_price_segment.png`
- `a4_building_age_by_price_segment.png`
- `a5_top_room_types.png`
- `a5_room_type_unit_price.png`
- `a6_orientation_unit_price.png`
- `a7_building_age_unit_price.png`
- `a8_municipality_vs_non_municipality.png`

## Key Learning Outcomes

This project helped me practise data cleaning and exploratory data analysis using Python. I learned how to transform messy text-based data into structured numerical features and use those features to generate useful market insights.

The project also helped me understand that data analysis results should be interpreted carefully. For example, city-level price comparisons may be affected by sample size, and correlation results should not be treated as causal evidence.

## Portfolio Relevance

This project provides evidence of my development in:

- Python data analysis
- data cleaning
- feature engineering
- exploratory data analysis
- data visualisation
- professional project documentation
