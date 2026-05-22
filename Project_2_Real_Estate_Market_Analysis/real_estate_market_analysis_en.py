"""
Real Estate Market Analysis and Value Evaluation Project

This script analyses a house sales dataset and produces cleaned data,
summary tables, charts, and a project summary for a professional learning
portfolio.

Main workflow:
1. Load the raw dataset
2. Inspect the dataset structure
3. Clean and transform raw text fields
4. Create useful analytical features
5. Generate summary tables
6. Generate visual evidence
7. Save all outputs for GitHub portfolio evidence
"""

from pathlib import Path
import warnings
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------
# 1. Project settings
# ---------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
DATA_FILE = PROJECT_DIR / "house_sales.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs_real_estate_project"
OUTPUT_DIR.mkdir(exist_ok=True)

ANALYSIS_YEAR = 2025

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


# ---------------------------------------------------------------------
# 2. Helper functions
# ---------------------------------------------------------------------
def extract_number(series: pd.Series) -> pd.Series:
    """Extract the first numeric value from a text-based pandas Series."""
    return pd.to_numeric(
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.extract(r"([-+]?\d*\.?\d+)")[0],
        errors="coerce",
    )


def parse_floor_level(value) -> str:
    """Convert floor text into a broader floor-level group."""
    text = str(value)
    if "浣�" in text:
        return "Low floor"
    if "涓�" in text:
        return "Middle floor"
    if "楂�" in text:
        return "High floor"
    return "Unknown"


def parse_orientation(value) -> str:
    """Convert orientation text into a broader orientation group."""
    text = str(value)
    if "鍗楀寳" in text:
        return "South-North"
    if "涓滆タ" in text:
        return "East-West"
    if "鍗�" in text:
        return "South"
    if "鍖�" in text:
        return "North"
    if "涓�" in text:
        return "East"
    if "瑗�" in text:
        return "West"
    return "Other/Unknown"


def remove_iqr_outliers(data: pd.DataFrame, column: str, factor: float = 2.0) -> pd.DataFrame:
    """Remove extreme outliers using a relaxed IQR rule."""
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1

    lower = q1 - factor * iqr
    upper = q3 + factor * iqr

    return data[(data[column] >= lower) & (data[column] <= upper)].copy()


def save_bar_chart(
    data: pd.Series,
    title: str,
    xlabel: str,
    ylabel: str,
    filename: str,
    rotation: int = 30,
) -> None:
    """Save a bar chart from a pandas Series."""
    plt.figure(figsize=(9, 5))
    data.plot(kind="bar")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=rotation, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300)
    plt.close()


def save_dataframe(df: pd.DataFrame, filename: str) -> None:
    """Save dataframe using UTF-8 with BOM so Chinese text opens correctly in Excel."""
    df.to_csv(OUTPUT_DIR / filename, index=False, encoding="utf-8-sig")


# ---------------------------------------------------------------------
# 3. Load and inspect the dataset
# ---------------------------------------------------------------------
if not DATA_FILE.exists():
    raise FileNotFoundError(
        "Cannot find house_sales.csv. Please place house_sales.csv "
        "in the same folder as this Python file."
    )

print("=" * 70)
print("Real Estate Market Analysis and Value Evaluation")
print("=" * 70)

df = pd.read_csv(DATA_FILE)

print("\nRaw dataset shape:")
print(df.shape)

print("\nFirst five rows:")
print(df.head())

raw_overview = pd.DataFrame(
    {
        "column": df.columns,
        "data_type": df.dtypes.astype(str).values,
        "missing_values": df.isna().sum().values,
        "unique_values": df.nunique().values,
    }
)
save_dataframe(raw_overview, "01_raw_dataset_overview.csv")


# ---------------------------------------------------------------------
# 4. Data cleaning and feature engineering
# ---------------------------------------------------------------------
clean_df = df.copy()

# Remove duplicated listing URLs first, because the same property may appear more than once.
rows_after_missing_check = len(clean_df.dropna(subset=["area", "price", "unit"]))

if "origin_url" in clean_df.columns:
    clean_df = clean_df.drop_duplicates(subset=["origin_url"]).copy()
else:
    clean_df = clean_df.drop_duplicates().copy()

rows_after_duplicate_removal = len(clean_df)

# Convert text-based numeric columns into analysis-ready numeric variables.
clean_df["area_m2"] = extract_number(clean_df["area"])
clean_df["total_price_10k_yuan"] = extract_number(clean_df["price"])
clean_df["unit_price_yuan_m2"] = extract_number(clean_df["unit"])

# Extract bedroom and living-room counts from room descriptions such as "3瀹�2鍘�".
clean_df["bedrooms"] = extract_number(
    clean_df["rooms"].astype(str).str.extract(r"(\d+)瀹�", expand=False)
)
clean_df["living_rooms"] = extract_number(
    clean_df["rooms"].astype(str).str.extract(r"(\d+)鍘�", expand=False)
)

# Extract building year and calculate building age.
clean_df["building_year"] = extract_number(
    clean_df["year"].astype(str).str.extract(r"(\d{4})", expand=False)
)
clean_df["building_age"] = ANALYSIS_YEAR - clean_df["building_year"]

# Create grouped categorical variables for analysis.
clean_df["floor_group"] = clean_df["floor"].apply(parse_floor_level)
clean_df["orientation_group"] = clean_df["toward"].apply(parse_orientation)

municipalities = ["鍖椾含", "涓婃捣", "澶╂触", "閲嶅簡"]
clean_df["city_type"] = np.where(
    clean_df["city"].isin(municipalities),
    "Municipality",
    "Non-municipality",
)

clean_df["room_type"] = (
    clean_df["bedrooms"].fillna(-1).astype(int).astype(str)
    + " bedrooms, "
    + clean_df["living_rooms"].fillna(-1).astype(int).astype(str)
    + " living rooms"
)
clean_df.loc[clean_df["room_type"].str.contains("-1"), "room_type"] = "Unknown"

# Keep rows with valid core numeric fields for market analysis.
analysis_df = clean_df.dropna(
    subset=["area_m2", "total_price_10k_yuan", "unit_price_yuan_m2"]
).copy()

# Remove clearly unreasonable values.
analysis_df = analysis_df[
    (analysis_df["area_m2"] >= 10)
    & (analysis_df["area_m2"] <= 500)
    & (analysis_df["total_price_10k_yuan"] > 0)
    & (analysis_df["total_price_10k_yuan"] <= 1000)
    & (analysis_df["unit_price_yuan_m2"] > 0)
    & (analysis_df["unit_price_yuan_m2"] <= 50000)
].copy()

# Remove extreme outliers using IQR.
analysis_df = remove_iqr_outliers(analysis_df, "unit_price_yuan_m2", factor=2.0)
analysis_df = remove_iqr_outliers(analysis_df, "total_price_10k_yuan", factor=2.0)

# Price segment based on unit price quartiles.
analysis_df["price_segment"] = pd.qcut(
    analysis_df["unit_price_yuan_m2"],
    q=4,
    labels=["Low price", "Mid price", "High price", "Luxury"],
    duplicates="drop",
)

# Building age groups for later analysis.
age_source = analysis_df["building_age"].astype("float")
analysis_df["building_age_group"] = pd.cut(
    age_source,
    bins=[0, 5, 10, 20, 30, 50, 100],
    labels=[
        "0-5 years",
        "6-10 years",
        "11-20 years",
        "21-30 years",
        "31-50 years",
        "51+ years",
    ],
    include_lowest=True,
)

cleaning_summary = pd.DataFrame(
    {
        "step": [
            "Raw rows",
            "Rows after dropping missing essential values",
            "Rows after removing duplicated listing URLs",
            "Rows after full cleaning and outlier filtering",
            "Rows removed in total",
        ],
        "row_count": [
            len(df),
            rows_after_missing_check,
            rows_after_duplicate_removal,
            len(analysis_df),
            len(df) - len(analysis_df),
        ],
    }
)
cleaning_summary["percentage_of_raw"] = (
    cleaning_summary["row_count"] / len(df) * 100
).round(2)

save_dataframe(cleaning_summary, "02_cleaning_summary.csv")
save_dataframe(analysis_df, "03_cleaned_house_sales.csv")

print("\nCleaning summary:")
print(cleaning_summary)

print("\nCleaned dataset shape:")
print(analysis_df.shape)


# ---------------------------------------------------------------------
# 5. Descriptive statistics
# ---------------------------------------------------------------------
numeric_columns = [
    "area_m2",
    "total_price_10k_yuan",
    "unit_price_yuan_m2",
    "bedrooms",
    "living_rooms",
    "building_age",
]

numeric_summary = analysis_df[numeric_columns].describe().T.round(2)
numeric_summary.insert(0, "variable", numeric_summary.index)
numeric_summary = numeric_summary.reset_index(drop=True)

save_dataframe(numeric_summary, "04_numeric_summary.csv")

print("\nNumeric summary:")
print(numeric_summary)


# ---------------------------------------------------------------------
# 6. Analysis A1: Correlation between key numerical variables
# ---------------------------------------------------------------------
correlation = analysis_df[numeric_columns].corr(numeric_only=True)
correlation.to_csv(OUTPUT_DIR / "05_correlation_matrix.csv", encoding="utf-8-sig")

plt.figure(figsize=(8, 6))
plt.imshow(correlation, aspect="auto")
plt.colorbar(label="Correlation")
plt.xticks(range(len(correlation.columns)), correlation.columns, rotation=45, ha="right")
plt.yticks(range(len(correlation.index)), correlation.index)

for i in range(len(correlation.index)):
    for j in range(len(correlation.columns)):
        plt.text(
            j,
            i,
            f"{correlation.iloc[i, j]:.2f}",
            ha="center",
            va="center",
        )

plt.title("Correlation Matrix of Key Housing Variables")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "a1_correlation_matrix.png", dpi=300)
plt.close()


# ---------------------------------------------------------------------
# 7. Analysis A2: Overall price distribution
# ---------------------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.hist(analysis_df["total_price_10k_yuan"], bins=40)
plt.title("Distribution of Total House Price")
plt.xlabel("Total price (10,000 yuan)")
plt.ylabel("Number of listings")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "a2_total_price_distribution.png", dpi=300)
plt.close()

plt.figure(figsize=(7, 5))
plt.boxplot(analysis_df["unit_price_yuan_m2"].dropna(), vert=True)
plt.title("Boxplot of Unit Price")
plt.ylabel("Unit price (yuan/m虏)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "a2_unit_price_boxplot.png", dpi=300)
plt.close()


# ---------------------------------------------------------------------
# 8. Analysis A3: City-level unit price comparison
# ---------------------------------------------------------------------
city_summary = (
    analysis_df.groupby("city")
    .agg(
        listing_count=("unit_price_yuan_m2", "count"),
        median_unit_price=("unit_price_yuan_m2", "median"),
        mean_unit_price=("unit_price_yuan_m2", "mean"),
        median_total_price_10k_yuan=("total_price_10k_yuan", "median"),
        median_area_m2=("area_m2", "median"),
    )
    .reset_index()
)

city_summary = city_summary.sort_values("median_unit_price", ascending=False)
save_dataframe(city_summary, "06_city_price_summary.csv")

top_cities = city_summary[city_summary["listing_count"] >= 30].head(10)

plt.figure(figsize=(9, 5))
plt.bar(top_cities["city"], top_cities["median_unit_price"])
plt.title("Top 10 Cities by Median Unit Price")
plt.xlabel("City")
plt.ylabel("Median unit price (yuan/m虏)")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "a3_top_city_unit_prices.png", dpi=300)
plt.close()

print("\nTop 10 cities by median unit price:")
print(top_cities)


# ---------------------------------------------------------------------
# 9. Analysis A4: Price segment analysis
# ---------------------------------------------------------------------
segment_summary = (
    analysis_df.groupby("price_segment", observed=False)
    .agg(
        listing_count=("unit_price_yuan_m2", "count"),
        median_area_m2=("area_m2", "median"),
        median_total_price_10k_yuan=("total_price_10k_yuan", "median"),
        median_unit_price=("unit_price_yuan_m2", "median"),
        median_bedrooms=("bedrooms", "median"),
        median_building_age=("building_age", "median"),
    )
    .reset_index()
)

save_dataframe(segment_summary, "07_price_segment_summary.csv")

save_bar_chart(
    segment_summary.set_index("price_segment")["median_area_m2"],
    "Median Area by Price Segment",
    "Price segment",
    "Median area (m虏)",
    "a4_median_area_by_price_segment.png",
)

save_bar_chart(
    segment_summary.set_index("price_segment")["median_building_age"],
    "Median Building Age by Price Segment",
    "Price segment",
    "Median building age (years)",
    "a4_building_age_by_price_segment.png",
)


# ---------------------------------------------------------------------
# 10. Analysis A5: Room type analysis
# ---------------------------------------------------------------------
room_counts = analysis_df["room_type"].value_counts().head(10)
room_counts.to_csv(OUTPUT_DIR / "08_room_type_counts.csv", encoding="utf-8-sig")

save_bar_chart(
    room_counts,
    "Top 10 Room Types by Listing Count",
    "Room type",
    "Number of listings",
    "a5_top_room_types.png",
)

top_room_types = room_counts.index.tolist()
room_price_summary = (
    analysis_df[analysis_df["room_type"].isin(top_room_types)]
    .groupby("room_type")
    .agg(
        listing_count=("unit_price_yuan_m2", "count"),
        median_total_price_10k_yuan=("total_price_10k_yuan", "median"),
        median_unit_price=("unit_price_yuan_m2", "median"),
    )
    .sort_values("median_unit_price", ascending=False)
)

room_price_summary.to_csv(
    OUTPUT_DIR / "09_room_type_price_summary.csv",
    encoding="utf-8-sig",
)

save_bar_chart(
    room_price_summary["median_unit_price"],
    "Median Unit Price by Common Room Type",
    "Room type",
    "Median unit price (yuan/m虏)",
    "a5_room_type_unit_price.png",
)


# ---------------------------------------------------------------------
# 11. Analysis A6: Orientation analysis
# ---------------------------------------------------------------------
orientation_summary = (
    analysis_df.groupby("orientation_group")
    .agg(
        listing_count=("unit_price_yuan_m2", "count"),
        median_unit_price=("unit_price_yuan_m2", "median"),
        median_total_price_10k_yuan=("total_price_10k_yuan", "median"),
    )
    .sort_values("median_unit_price", ascending=False)
)

orientation_summary.to_csv(
    OUTPUT_DIR / "10_orientation_price_summary.csv",
    encoding="utf-8-sig",
)

save_bar_chart(
    orientation_summary["median_unit_price"],
    "Median Unit Price by Home Orientation",
    "Orientation",
    "Median unit price (yuan/m虏)",
    "a6_orientation_unit_price.png",
)


# ---------------------------------------------------------------------
# 12. Analysis A7: Building age analysis
# ---------------------------------------------------------------------
age_summary = (
    analysis_df.dropna(subset=["building_age_group"])
    .groupby("building_age_group", observed=False)
    .agg(
        listing_count=("unit_price_yuan_m2", "count"),
        median_unit_price=("unit_price_yuan_m2", "median"),
        median_total_price_10k_yuan=("total_price_10k_yuan", "median"),
    )
)

age_summary.to_csv(
    OUTPUT_DIR / "11_building_age_price_summary.csv",
    encoding="utf-8-sig",
)

save_bar_chart(
    age_summary["median_unit_price"],
    "Median Unit Price by Building Age Group",
    "Building age group",
    "Median unit price (yuan/m虏)",
    "a7_building_age_unit_price.png",
)


# ---------------------------------------------------------------------
# 13. Analysis A8: Municipality and non-municipality comparison
# ---------------------------------------------------------------------
city_type_summary = (
    analysis_df.groupby("city_type")
    .agg(
        listing_count=("unit_price_yuan_m2", "count"),
        median_unit_price=("unit_price_yuan_m2", "median"),
        median_total_price_10k_yuan=("total_price_10k_yuan", "median"),
        median_area_m2=("area_m2", "median"),
    )
)

city_type_summary.to_csv(
    OUTPUT_DIR / "12_city_type_summary.csv",
    encoding="utf-8-sig",
)

save_bar_chart(
    city_type_summary["median_unit_price"],
    "Median Unit Price: Municipality vs Non-municipality",
    "City type",
    "Median unit price (yuan/m虏)",
    "a8_municipality_vs_non_municipality.png",
    rotation=0,
)


# ---------------------------------------------------------------------
# 14. Final project summary
# ---------------------------------------------------------------------
summary_path = OUTPUT_DIR / "13_project_summary.txt"

with open(summary_path, "w", encoding="utf-8") as f:
    f.write("Real Estate Market Analysis Project Summary\n")
    f.write("===========================================\n\n")
    f.write(f"Raw dataset shape: {df.shape}\n")
    f.write(f"Rows after missing essential value check: {rows_after_missing_check}\n")
    f.write(f"Rows after duplicate removal: {rows_after_duplicate_removal}\n")
    f.write(f"Rows used for final analysis: {len(analysis_df)}\n\n")

    f.write("Main analysis questions:\n")
    f.write("A1. Which variables are most related to house price?\n")
    f.write("A2. What is the overall price distribution?\n")
    f.write("A3. Which cities have the highest unit prices?\n")
    f.write("A4. What features are common in different price segments?\n")
    f.write("A5. Which room types are most common and how do their prices differ?\n")
    f.write("A6. Are south-facing or south-north-facing homes more expensive?\n")
    f.write("A7. How does building age relate to price?\n")
    f.write("A8. Are municipality homes more expensive than non-municipality homes?\n\n")

    f.write("Key evidence files generated:\n")
    f.write("- 01_raw_dataset_overview.csv\n")
    f.write("- 02_cleaning_summary.csv\n")
    f.write("- 03_cleaned_house_sales.csv\n")
    f.write("- 04_numeric_summary.csv\n")
    f.write("- 05_correlation_matrix.csv\n")
    f.write("- 06_city_price_summary.csv\n")
    f.write("- 07_price_segment_summary.csv\n")
    f.write("- 08_room_type_counts.csv\n")
    f.write("- 09_room_type_price_summary.csv\n")
    f.write("- 10_orientation_price_summary.csv\n")
    f.write("- 11_building_age_price_summary.csv\n")
    f.write("- 12_city_type_summary.csv\n")
    f.write("- 13_project_summary.txt\n")
    f.write("- a1_correlation_matrix.png\n")
    f.write("- a2_total_price_distribution.png\n")
    f.write("- a3_top_city_unit_prices.png\n")
    f.write("- a8_municipality_vs_non_municipality.png\n\n")

    f.write("Portfolio reflection note:\n")
    f.write(
        "This project demonstrates a complete exploratory data analysis workflow. "
        "It shows how raw housing listing data can be cleaned, transformed, grouped, "
        "visualised, and interpreted to support real estate market insight. The project "
        "also demonstrates practical use of Python, pandas, and matplotlib for data analytics."
    )

print("\nProject completed successfully.")
print(f"All output files have been saved in: {OUTPUT_DIR}")
