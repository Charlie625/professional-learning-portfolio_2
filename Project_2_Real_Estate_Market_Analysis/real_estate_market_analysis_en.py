import os
import re
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================
# Project Settings
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
DATA_FILE = PROJECT_DIR / "house_sales.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs_real_estate_project"

CURRENT_YEAR = datetime.now().year

OUTPUT_DIR.mkdir(exist_ok=True)

plt.rcParams["font.sans-serif"] = [
    "SimHei",
    "Microsoft YaHei",
    "Arial Unicode MS",
    "DejaVu Sans"
]
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# Helper Functions
# ============================================================

def clean_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def extract_first_number(value):
    text = clean_text(value).replace(",", "")
    match = re.search(r"\d+(?:\.\d+)?", text)
    if match:
        return float(match.group())
    return np.nan


def parse_rooms(room_text):
    text = clean_text(room_text)

    bedroom_match = re.search(r"(\d+)\s*室", text)
    living_room_match = re.search(r"(\d+)\s*厅", text)

    bedrooms = float(bedroom_match.group(1)) if bedroom_match else np.nan
    living_rooms = float(living_room_match.group(1)) if living_room_match else np.nan

    return pd.Series([bedrooms, living_rooms])


def parse_build_year(year_text):
    text = clean_text(year_text)
    match = re.search(r"(19|20)\d{2}", text)
    if match:
        return float(match.group())
    return np.nan


def parse_floor_level(floor_text):
    text = clean_text(floor_text)

    if "低" in text:
        return "Low floor"
    if "中" in text:
        return "Middle floor"
    if "高" in text:
        return "High floor"
    if "顶" in text:
        return "Top floor"
    if "底" in text:
        return "Ground floor"

    return "Unknown"


def parse_total_floor(floor_text):
    text = clean_text(floor_text)
    match = re.search(r"共\s*(\d+)\s*层", text)

    if match:
        return float(match.group(1))

    numbers = re.findall(r"\d+", text)
    if numbers:
        return float(numbers[-1])

    return np.nan


def normalise_orientation(toward_text):
    text = clean_text(toward_text)

    if text == "":
        return "Other/Unknown"

    if "南北" in text or ("南" in text and "北" in text):
        return "South-North"
    if "东西" in text or ("东" in text and "西" in text):
        return "East-West"
    if "南" in text:
        return "South"
    if "北" in text:
        return "North"
    if "东" in text:
        return "East"
    if "西" in text:
        return "West"

    return "Other/Unknown"


def make_room_type(row):
    bedrooms = row["bedrooms"]
    living_rooms = row["living_rooms"]

    if pd.isna(bedrooms) or pd.isna(living_rooms):
        return "Unknown"

    return f"{int(bedrooms)} bedrooms, {int(living_rooms)} living rooms"


def classify_city_type(city):
    municipality_cities = ["北京", "上海", "天津", "重庆"]

    if clean_text(city) in municipality_cities:
        return "Municipality"
    return "Non-municipality"


def translate_city_name(city):
    city_map = {
        "北京": "Beijing",
        "上海": "Shanghai",
        "天津": "Tianjin",
        "重庆": "Chongqing",
        "南京": "Nanjing",
        "苏州": "Suzhou",
        "无锡": "Wuxi",
        "常州": "Changzhou",
        "南通": "Nantong",
        "徐州": "Xuzhou",
        "扬州": "Yangzhou",
        "镇江": "Zhenjiang",
        "盐城": "Yancheng",
        "泰州": "Taizhou",
        "宿迁": "Suqian",
        "杭州": "Hangzhou",
        "宁波": "Ningbo",
        "温州": "Wenzhou",
        "嘉兴": "Jiaxing",
        "绍兴": "Shaoxing",
        "金华": "Jinhua",
        "福州": "Fuzhou",
        "厦门": "Xiamen",
        "泉州": "Quanzhou",
        "漳州": "Zhangzhou",
        "莆田": "Putian",
        "三明": "Sanming",
        "南平": "Nanping",
        "龙岩": "Longyan",
        "宁德": "Ningde",
        "合肥": "Hefei",
        "芜湖": "Wuhu",
        "蚌埠": "Bengbu",
        "淮南": "Huainan",
        "淮北": "Huaibei",
        "马鞍山": "Maanshan",
        "安庆": "Anqing",
        "黄山": "Huangshan",
        "阜阳": "Fuyang",
        "宿州": "Suzhou Anhui",
        "滁州": "Chuzhou",
        "六安": "Luan",
        "亳州": "Bozhou",
        "池州": "Chizhou",
        "宣城": "Xuancheng",
        "广州": "Guangzhou",
        "深圳": "Shenzhen",
        "佛山": "Foshan",
        "东莞": "Dongguan",
        "珠海": "Zhuhai",
        "中山": "Zhongshan",
        "成都": "Chengdu",
        "武汉": "Wuhan",
        "郑州": "Zhengzhou",
        "长沙": "Changsha",
        "西安": "Xi'an",
        "济南": "Jinan",
        "青岛": "Qingdao",
        "沈阳": "Shenyang",
        "大连": "Dalian",
        "昆明": "Kunming",
        "南昌": "Nanchang",
        "南宁": "Nanning",
        "海口": "Haikou",
        "三亚": "Sanya",
        "平潭": "Pingtan",
        "长乐": "Changle",
        "连江": "Lianjiang",
        "罗源": "Luoyuan",
        "闽侯": "Minhou",
        "闽清": "Minqing",
        "永泰": "Yongtai",
    }

    city = clean_text(city)
    return city_map.get(city, city)


def save_table(dataframe, file_name):
    dataframe.to_csv(OUTPUT_DIR / file_name, index=False, encoding="utf-8-sig")


def add_value_labels(ax):
    for bar in ax.patches:
        height = bar.get_height()
        if pd.notna(height):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.0f}",
                ha="center",
                va="bottom",
                fontsize=8
            )


def save_bar_chart(dataframe, x_col, y_col, title, x_label, y_label, file_name, rotation=30):
    if dataframe.empty:
        print(f"Skipped {file_name}: no data available.")
        return

    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    ax.bar(dataframe[x_col].astype(str), dataframe[y_col])
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.xticks(rotation=rotation, ha="right")
    add_value_labels(ax)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / file_name, dpi=300)
    plt.close()


# ============================================================
# 1. Load Dataset
# ============================================================

print("=" * 70)
print("Real Estate Market Analysis and Value Evaluation")
print("=" * 70)

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Cannot find {DATA_FILE.name}. Please place house_sales.csv in the same folder as this Python file."
    )

df_raw = pd.read_csv(DATA_FILE)

print("\nRaw dataset shape:")
print(df_raw.shape)

print("\nFirst five rows:")
print(df_raw.head())


required_columns = [
    "city",
    "address",
    "area",
    "floor",
    "name",
    "price",
    "province",
    "rooms",
    "toward",
    "unit",
    "year",
    "origin_url"
]

missing_columns = [col for col in required_columns if col not in df_raw.columns]

if missing_columns:
    raise ValueError(f"The following required columns are missing: {missing_columns}")


# ============================================================
# 2. Raw Dataset Overview
# ============================================================

raw_overview = pd.DataFrame({
    "column": df_raw.columns,
    "data_type": [str(df_raw[col].dtype) for col in df_raw.columns],
    "missing_values": [df_raw[col].isna().sum() for col in df_raw.columns],
    "unique_values": [df_raw[col].nunique(dropna=True) for col in df_raw.columns]
})

save_table(raw_overview, "01_raw_dataset_overview.csv")


# ============================================================
# 3. Data Cleaning and Feature Engineering
# ============================================================

df = df_raw.copy()

for col in df.columns:
    if df[col].dtype == "object" or pd.api.types.is_string_dtype(df[col]):
        df[col] = df[col].map(clean_text)

df["area_m2"] = df["area"].map(extract_first_number)
df["total_price_10k_yuan"] = df["price"].map(extract_first_number)
df["unit_price_yuan_m2"] = df["unit"].map(extract_first_number)

df[["bedrooms", "living_rooms"]] = df["rooms"].apply(parse_rooms)

df["build_year"] = df["year"].map(parse_build_year)
df["building_age"] = CURRENT_YEAR - df["build_year"]

df["floor_level"] = df["floor"].map(parse_floor_level)
df["total_floors"] = df["floor"].map(parse_total_floor)

df["orientation_group"] = df["toward"].map(normalise_orientation)
df["room_type"] = df.apply(make_room_type, axis=1)
df["city_type"] = df["city"].map(classify_city_type)
df["city_label"] = df["city"].map(translate_city_name)


essential_columns = [
    "city",
    "province",
    "area_m2",
    "total_price_10k_yuan",
    "unit_price_yuan_m2",
    "bedrooms",
    "living_rooms"
]

df_after_missing = df.dropna(subset=essential_columns).copy()
if "origin_url" in df_after_missing.columns:
    df_after_duplicates = df_after_missing.drop_duplicates(subset=["origin_url"]).copy()
else:
    df_after_duplicates = df_after_missing.drop_duplicates().copy()

df_clean = df_after_duplicates[
    (df_after_duplicates["area_m2"].between(10, 500)) &
    (df_after_duplicates["total_price_10k_yuan"].between(1, 1000)) &
    (df_after_duplicates["unit_price_yuan_m2"].between(500, 50000)) &
    (df_after_duplicates["bedrooms"].between(0, 10)) &
    (df_after_duplicates["living_rooms"].between(0, 12))
].copy()

if df_clean.empty:
    raise ValueError("No rows remain after cleaning. Please check the cleaning rules or the input dataset.")

# Important fix for your error:
# Convert building_age into normal float values before using pd.cut.
df_clean["building_age"] = pd.to_numeric(
    df_clean["building_age"],
    errors="coerce"
).astype("float64")


building_age_bins = [-np.inf, 5, 10, 20, 30, 50, np.inf]
building_age_labels = [
    "0-5 years",
    "6-10 years",
    "11-20 years",
    "21-30 years",
    "31-50 years",
    "51+ years"
]

df_clean["building_age_group"] = pd.cut(
    df_clean["building_age"],
    bins=building_age_bins,
    labels=building_age_labels,
    include_lowest=True
)

df_clean["building_age_group"] = (
    df_clean["building_age_group"]
    .astype("object")
    .fillna("Unknown")
)


area_bins = [0, 70, 90, 120, 150, 200, np.inf]
area_labels = [
    "0-70 m2",
    "71-90 m2",
    "91-120 m2",
    "121-150 m2",
    "151-200 m2",
    "200+ m2"
]

df_clean["area_group"] = pd.cut(
    df_clean["area_m2"],
    bins=area_bins,
    labels=area_labels,
    include_lowest=True
)


price_q1 = df_clean["total_price_10k_yuan"].quantile(0.25)
price_q2 = df_clean["total_price_10k_yuan"].quantile(0.50)
price_q3 = df_clean["total_price_10k_yuan"].quantile(0.75)


def classify_price_segment(value):
    if pd.isna(value):
        return "Unknown"
    if value <= price_q1:
        return "Low price"
    if value <= price_q2:
        return "Mid price"
    if value <= price_q3:
        return "High price"
    return "Luxury"


df_clean["price_segment"] = df_clean["total_price_10k_yuan"].map(classify_price_segment)


cleaning_summary = pd.DataFrame({
    "step": [
        "Raw rows",
        "Rows after dropping missing essential values",
        "Rows after removing duplicated listing URLs",
        "Rows after full cleaning and outlier filtering",
        "Rows removed in total"
    ],
    "row_count": [
        len(df_raw),
        len(df_after_missing),
        len(df_after_duplicates),
        len(df_clean),
        len(df_raw) - len(df_clean)
    ]
})

cleaning_summary["percentage_of_raw"] = (
    cleaning_summary["row_count"] / len(df_raw) * 100
).round(2)

save_table(cleaning_summary, "02_cleaning_summary.csv")
df_clean.to_csv(OUTPUT_DIR / "03_cleaned_house_sales.csv", index=False, encoding="utf-8-sig")

print("\nCleaning summary:")
print(cleaning_summary)

print("\nCleaned dataset shape:")
print(df_clean.shape)


# ============================================================
# 4. Descriptive Statistics
# ============================================================

numeric_columns = [
    "area_m2",
    "total_price_10k_yuan",
    "unit_price_yuan_m2",
    "bedrooms",
    "living_rooms",
    "building_age"
]

numeric_summary = (
    df_clean[numeric_columns]
    .describe(percentiles=[0.25, 0.5, 0.75])
    .T
    .reset_index()
    .rename(columns={"index": "variable"})
)

numeric_summary = numeric_summary.round(2)
save_table(numeric_summary, "04_numeric_summary.csv")

print("\nNumeric summary:")
print(numeric_summary)


# ============================================================
# 5. Correlation Analysis
# ============================================================

correlation_columns = [
    "area_m2",
    "total_price_10k_yuan",
    "unit_price_yuan_m2",
    "bedrooms",
    "living_rooms",
    "building_age"
]

correlation_matrix = df_clean[correlation_columns].corr(numeric_only=True).round(3)
correlation_matrix.to_csv(OUTPUT_DIR / "05_correlation_matrix.csv", encoding="utf-8-sig")

plt.figure(figsize=(12, 9))
ax = plt.gca()

image = ax.imshow(correlation_matrix, vmin=-1, vmax=1)

ax.set_xticks(range(len(correlation_matrix.columns)))
ax.set_yticks(range(len(correlation_matrix.index)))
ax.set_xticklabels(correlation_matrix.columns, rotation=45, ha="right")
ax.set_yticklabels(correlation_matrix.index)

for i in range(len(correlation_matrix.index)):
    for j in range(len(correlation_matrix.columns)):
        value = correlation_matrix.iloc[i, j]
        ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=9)

plt.colorbar(image, ax=ax, label="Correlation")
plt.title("Correlation Matrix of Key Housing Variables")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "a1_correlation_matrix.png", dpi=300)
plt.close()


# ============================================================
# 6. Price Distribution and Outliers
# ============================================================

plt.figure(figsize=(10, 6))
plt.hist(df_clean["total_price_10k_yuan"], bins=40)
plt.title("Distribution of Total House Price")
plt.xlabel("Total price (10,000 yuan)")
plt.ylabel("Number of listings")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "a2_total_price_distribution.png", dpi=300)
plt.close()


plt.figure(figsize=(8, 6))
plt.boxplot(df_clean["unit_price_yuan_m2"].dropna())
plt.title("Boxplot of Unit Price")
plt.ylabel("Unit price (yuan/m2)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "a2_unit_price_boxplot.png", dpi=300)
plt.close()


# ============================================================
# 7. City Comparison
# ============================================================

city_price_summary = (
    df_clean
    .groupby(["city", "city_label", "province"], as_index=False)
    .agg(
        listing_count=("city", "size"),
        median_unit_price_yuan_m2=("unit_price_yuan_m2", "median"),
        mean_unit_price_yuan_m2=("unit_price_yuan_m2", "mean"),
        median_total_price_10k_yuan=("total_price_10k_yuan", "median"),
        median_area_m2=("area_m2", "median")
    )
)

city_price_summary = city_price_summary.round(2)
save_table(city_price_summary, "06_city_price_summary.csv")

city_summary_for_chart = city_price_summary[
    city_price_summary["listing_count"] >= 30
].copy()

top_city_unit_prices = (
    city_summary_for_chart
    .sort_values("median_unit_price_yuan_m2", ascending=False)
    .head(10)
)

print("\nTop 10 cities by median unit price:")
print(top_city_unit_prices)

save_bar_chart(
    top_city_unit_prices,
    "city_label",
    "median_unit_price_yuan_m2",
    "Top 10 Cities by Median Unit Price",
    "City",
    "Median unit price (yuan/m2)",
    "a3_top_city_unit_prices.png"
)


# ============================================================
# 8. Price Segment Analysis
# ============================================================

price_segment_order = ["Low price", "Mid price", "High price", "Luxury"]

price_segment_summary = (
    df_clean
    .groupby("price_segment", as_index=False)
    .agg(
        listing_count=("price_segment", "size"),
        median_total_price_10k_yuan=("total_price_10k_yuan", "median"),
        median_unit_price_yuan_m2=("unit_price_yuan_m2", "median"),
        median_area_m2=("area_m2", "median"),
        median_building_age=("building_age", "median"),
        median_bedrooms=("bedrooms", "median")
    )
)

price_segment_summary["price_segment"] = pd.Categorical(
    price_segment_summary["price_segment"],
    categories=price_segment_order,
    ordered=True
)

price_segment_summary = (
    price_segment_summary
    .sort_values("price_segment")
    .round(2)
)

save_table(price_segment_summary, "07_price_segment_summary.csv")


save_bar_chart(
    price_segment_summary,
    "price_segment",
    "median_area_m2",
    "Median Area by Price Segment",
    "Price segment",
    "Median area (m2)",
    "a4_median_area_by_price_segment.png"
)

age_segment_data = price_segment_summary.dropna(subset=["median_building_age"])

save_bar_chart(
    age_segment_data,
    "price_segment",
    "median_building_age",
    "Median Building Age by Price Segment",
    "Price segment",
    "Median building age (years)",
    "a4_building_age_by_price_segment.png"
)


# ============================================================
# 9. Room Type Analysis
# ============================================================

room_type_counts = (
    df_clean["room_type"]
    .value_counts()
    .reset_index()
)

room_type_counts.columns = ["room_type", "listing_count"]
save_table(room_type_counts, "08_room_type_counts.csv")

top_room_types = room_type_counts.head(10)

save_bar_chart(
    top_room_types,
    "room_type",
    "listing_count",
    "Top 10 Room Types by Listing Count",
    "Room type",
    "Number of listings",
    "a5_top_room_types.png",
    rotation=35
)


room_type_price_summary = (
    df_clean
    .groupby("room_type", as_index=False)
    .agg(
        listing_count=("room_type", "size"),
        median_unit_price_yuan_m2=("unit_price_yuan_m2", "median"),
        median_total_price_10k_yuan=("total_price_10k_yuan", "median"),
        median_area_m2=("area_m2", "median")
    )
)

room_type_price_summary = (
    room_type_price_summary
    .sort_values("listing_count", ascending=False)
    .round(2)
)

save_table(room_type_price_summary, "09_room_type_price_summary.csv")

top_common_room_prices = (
    room_type_price_summary
    .head(10)
    .sort_values("median_unit_price_yuan_m2", ascending=False)
)

save_bar_chart(
    top_common_room_prices,
    "room_type",
    "median_unit_price_yuan_m2",
    "Median Unit Price by Common Room Type",
    "Room type",
    "Median unit price (yuan/m2)",
    "a5_room_type_unit_price.png",
    rotation=35
)


# ============================================================
# 10. Orientation Premium Analysis
# ============================================================

orientation_summary = (
    df_clean
    .groupby("orientation_group", as_index=False)
    .agg(
        listing_count=("orientation_group", "size"),
        median_unit_price_yuan_m2=("unit_price_yuan_m2", "median"),
        median_total_price_10k_yuan=("total_price_10k_yuan", "median"),
        median_area_m2=("area_m2", "median")
    )
)

orientation_summary = (
    orientation_summary
    .sort_values("median_unit_price_yuan_m2", ascending=False)
    .round(2)
)

save_table(orientation_summary, "10_orientation_price_summary.csv")

save_bar_chart(
    orientation_summary,
    "orientation_group",
    "median_unit_price_yuan_m2",
    "Median Unit Price by Home Orientation",
    "Orientation",
    "Median unit price (yuan/m2)",
    "a6_orientation_unit_price.png"
)


# ============================================================
# 11. Building Age Effect
# ============================================================

building_age_summary = (
    df_clean[df_clean["building_age_group"] != "Unknown"]
    .groupby("building_age_group", as_index=False)
    .agg(
        listing_count=("building_age_group", "size"),
        median_unit_price_yuan_m2=("unit_price_yuan_m2", "median"),
        median_total_price_10k_yuan=("total_price_10k_yuan", "median"),
        median_area_m2=("area_m2", "median")
    )
)

building_age_summary["building_age_group"] = pd.Categorical(
    building_age_summary["building_age_group"],
    categories=building_age_labels,
    ordered=True
)

building_age_summary = (
    building_age_summary
    .sort_values("building_age_group")
    .round(2)
)

save_table(building_age_summary, "11_building_age_price_summary.csv")

save_bar_chart(
    building_age_summary,
    "building_age_group",
    "median_unit_price_yuan_m2",
    "Median Unit Price by Building Age Group",
    "Building age group",
    "Median unit price (yuan/m2)",
    "a7_building_age_unit_price.png"
)


# ============================================================
# 12. Municipality vs Non-municipality Comparison
# ============================================================

city_type_summary = (
    df_clean
    .groupby("city_type", as_index=False)
    .agg(
        listing_count=("city_type", "size"),
        median_unit_price_yuan_m2=("unit_price_yuan_m2", "median"),
        median_total_price_10k_yuan=("total_price_10k_yuan", "median"),
        median_area_m2=("area_m2", "median")
    )
)

city_type_summary = city_type_summary.round(2)
save_table(city_type_summary, "12_city_type_summary.csv")

save_bar_chart(
    city_type_summary,
    "city_type",
    "median_unit_price_yuan_m2",
    "Municipality vs Non-municipality Median Unit Price",
    "City type",
    "Median unit price (yuan/m2)",
    "a8_municipality_vs_non_municipality.png",
    rotation=0
)


# ============================================================
# 13. Project Summary File
# ============================================================

strongest_price_correlations = (
    correlation_matrix["total_price_10k_yuan"]
    .drop(labels=["total_price_10k_yuan"])
    .abs()
    .sort_values(ascending=False)
)

summary_file = OUTPUT_DIR / "13_project_summary.txt"

with open(summary_file, "w", encoding="utf-8") as f:
    f.write("Real Estate Market Analysis and Value Evaluation\n")
    f.write("=" * 55 + "\n\n")

    f.write("1. Project Purpose\n")
    f.write(
        "This project analyses a real estate dataset to understand housing price "
        "distribution, key price-related variables, city differences, room type patterns, "
        "orientation effects, and building age effects.\n\n"
    )

    f.write("2. Dataset Information\n")
    f.write(f"Raw dataset shape: {df_raw.shape}\n")
    f.write(f"Cleaned dataset shape: {df_clean.shape}\n")
    f.write(f"Current year used for building age calculation: {CURRENT_YEAR}\n\n")

    f.write("3. Cleaning Method\n")
    f.write(
        "The cleaning process extracted numeric values from area, total price, "
        "unit price, room type, floor information, and building year. Rows with missing "
        "essential fields were removed. Duplicate listing URLs were removed. Extreme or "
        "unrealistic values were filtered using reasonable value ranges.\n\n"
    )

    f.write("4. Key Descriptive Statistics\n")
    f.write(numeric_summary.to_string(index=False))
    f.write("\n\n")

    f.write("5. Strongest Correlations with Total House Price\n")
    for variable, value in strongest_price_correlations.head(5).items():
        f.write(f"- {variable}: {value:.3f}\n")
    f.write("\n")

    f.write("6. Top Cities by Median Unit Price\n")
    f.write(top_city_unit_prices.to_string(index=False))
    f.write("\n\n")

    f.write("7. Price Segment Summary\n")
    f.write(price_segment_summary.to_string(index=False))
    f.write("\n\n")

    f.write("8. Room Type Summary\n")
    f.write(room_type_price_summary.head(10).to_string(index=False))
    f.write("\n\n")

    f.write("9. Orientation Summary\n")
    f.write(orientation_summary.to_string(index=False))
    f.write("\n\n")

    f.write("10. Reflection\n")
    f.write(
        "This project demonstrates a complete data analysis workflow using Python. "
        "It includes data loading, cleaning, feature engineering, descriptive statistics, "
        "correlation analysis, grouped comparison, data visualisation, and written evidence. "
        "The analysis is suitable for demonstrating practical skills in pandas and matplotlib."
    )


# ============================================================
# 14. Final Output
# ============================================================

print("\nProject completed successfully.")
print(f"All output files have been saved in: {OUTPUT_DIR}")

print("\nGenerated key files:")

generated_files = [
    "01_raw_dataset_overview.csv",
    "02_cleaning_summary.csv",
    "03_cleaned_house_sales.csv",
    "04_numeric_summary.csv",
    "05_correlation_matrix.csv",
    "06_city_price_summary.csv",
    "07_price_segment_summary.csv",
    "08_room_type_counts.csv",
    "09_room_type_price_summary.csv",
    "10_orientation_price_summary.csv",
    "11_building_age_price_summary.csv",
    "12_city_type_summary.csv",
    "13_project_summary.txt",
    "a1_correlation_matrix.png",
    "a2_total_price_distribution.png",
    "a2_unit_price_boxplot.png",
    "a3_top_city_unit_prices.png",
    "a4_median_area_by_price_segment.png",
    "a4_building_age_by_price_segment.png",
    "a5_top_room_types.png",
    "a5_room_type_unit_price.png",
    "a6_orientation_unit_price.png",
    "a7_building_age_unit_price.png",
    "a8_municipality_vs_non_municipality.png"
]

for file_name in generated_files:
    print(f"- {file_name}")
