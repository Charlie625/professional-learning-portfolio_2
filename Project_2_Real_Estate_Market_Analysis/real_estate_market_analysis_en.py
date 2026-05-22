"""
Real Estate Market Analysis and Value Evaluation

This script cleans and analyses a real estate listing dataset. It produces
CSV summaries, charts, and a written project summary as evidence for a
professional learning portfolio.
"""

from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
DATA_FILE = PROJECT_DIR / "house_sales.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs_real_estate_project"
CURRENT_YEAR = 2026

OUTPUT_DIR.mkdir(exist_ok=True)

CITY_NAME_MAP = {
    "北京": "Beijing",
    "上海": "Shanghai",
    "广州": "Guangzhou",
    "深圳": "Shenzhen",
    "杭州": "Hangzhou",
    "南京": "Nanjing",
    "苏州": "Suzhou",
    "成都": "Chengdu",
    "重庆": "Chongqing",
    "武汉": "Wuhan",
    "天津": "Tianjin",
    "厦门": "Xiamen",
    "福州": "Fuzhou",
    "宁波": "Ningbo",
    "无锡": "Wuxi",
    "合肥": "Hefei",
    "长沙": "Changsha",
    "郑州": "Zhengzhou",
    "西安": "Xi'an",
    "青岛": "Qingdao",
    "大连": "Dalian",
    "三亚": "Sanya",
    "佛山": "Foshan",
    "东莞": "Dongguan",
    "泉州": "Quanzhou",
    "常州": "Changzhou",
    "南通": "Nantong",
    "温州": "Wenzhou",
    "绍兴": "Shaoxing",
    "金华": "Jinhua",
    "嘉兴": "Jiaxing",
    "台州": "Taizhou",
    "永泰": "Yongtai",
    "长乐": "Changle",
    "连江": "Lianjiang",
    "罗源": "Luoyuan",
    "闽清": "Minqing",
    "平潭": "Pingtan",
    "镇海": "Zhenhai",
    "建德": "Jiande",
    "淳安": "Chun'an",
    "乐清": "Yueqing",
}

MUNICIPALITIES = {"北京", "上海", "天津", "重庆"}


def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first matching column name from a list of possible names."""
    for name in candidates:
        if name in df.columns:
            return name
    return None


def extract_first_number(value) -> float:
    """Extract the first numeric value from a text field."""
    if pd.isna(value):
        return np.nan
    text = str(value).replace(",", "")
    match = re.search(r"\d+(?:\.\d+)?", text)
    if match:
        return float(match.group())
    return np.nan


def extract_room_counts(value) -> tuple[float, float]:
    """Extract bedroom and living room counts from Chinese room type text."""
    if pd.isna(value):
        return np.nan, np.nan
    text = str(value)
    match = re.search(r"(\d+)\s*室\s*(\d+)\s*厅", text)
    if match:
        return float(match.group(1)), float(match.group(2))
    match = re.search(r"(\d+)\s*bedroom.*?(\d+)\s*living", text, flags=re.IGNORECASE)
    if match:
        return float(match.group(1)), float(match.group(2))
    return np.nan, np.nan


def extract_year(value) -> float:
    """Extract a four-digit building year."""
    if pd.isna(value):
        return np.nan
    text = str(value)
    match = re.search(r"(19\d{2}|20\d{2})", text)
    if match:
        year = int(match.group(1))
        if 1900 <= year <= CURRENT_YEAR:
            return float(year)
    return np.nan


def simplify_orientation(value) -> str:
    """Convert raw orientation text into a smaller set of English labels."""
    if pd.isna(value):
        return "Other/Unknown"
    text = str(value)
    if "南北" in text:
        return "South-North"
    if "南" in text:
        return "South"
    if "东" in text and "西" in text:
        return "East-West"
    if "东" in text:
        return "East"
    if "西" in text:
        return "West"
    if "北" in text:
        return "North"
    return "Other/Unknown"


def create_room_type_label(bedrooms, living_rooms) -> str:
    """Create an English room type label."""
    if pd.isna(bedrooms) or pd.isna(living_rooms):
        return "Unknown"
    return f"{int(bedrooms)} bedrooms, {int(living_rooms)} living rooms"


def save_bar_chart(series: pd.Series, title: str, xlabel: str, ylabel: str, filename: str, figsize=(10, 6)) -> None:
    """Save a simple bar chart."""
    plt.figure(figsize=figsize)
    plt.bar(series.index.astype(str), series.values)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300)
    plt.close()


def main() -> None:
    print("=" * 70)
    print("Real Estate Market Analysis and Value Evaluation")
    print("=" * 70)

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            "Cannot find house_sales.csv. Please place house_sales.csv in the same folder as this script."
        )

    df_raw = pd.read_csv(DATA_FILE)

    print("\nRaw dataset shape:")
    print(df_raw.shape)

    print("\nFirst five rows:")
    print(df_raw.head())

    raw_overview = pd.DataFrame(
        {
            "column": df_raw.columns,
            "data_type": df_raw.dtypes.astype(str).values,
            "missing_values": df_raw.isna().sum().values,
            "unique_values": df_raw.nunique(dropna=True).values,
        }
    )
    raw_overview.to_csv(OUTPUT_DIR / "01_raw_dataset_overview.csv", index=False)

    city_col = find_column(df_raw, ["city", "城市"])
    district_col = find_column(df_raw, ["district", "区域", "area_name"])
    address_col = find_column(df_raw, ["address", "地址"])
    area_col = find_column(df_raw, ["area", "建筑面积", "area_m2"])
    total_price_col = find_column(df_raw, ["total_price", "price", "总价", "total_price_10k_yuan"])
    unit_price_col = find_column(df_raw, ["unit_price", "unit", "单价", "unit_price_yuan_m2"])
    room_col = find_column(df_raw, ["rooms", "house_type", "户型", "layout"])
    floor_col = find_column(df_raw, ["floor", "楼层"])
    year_col = find_column(df_raw, ["year", "building_year", "建造年份", "built_year"])
    orientation_col = find_column(df_raw, ["orientation", "toward", "朝向"])
    url_col = find_column(df_raw, ["origin_url", "url", "link"])

    required_columns = [city_col, area_col, total_price_col, unit_price_col]
    if any(col is None for col in required_columns):
        raise ValueError(
            "The dataset does not contain the required columns for city, area, total price, and unit price."
        )

    df = df_raw.copy()

    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].astype("string").str.strip()

    df = df.dropna(subset=[city_col, area_col, total_price_col, unit_price_col])
    rows_after_essential = len(df)

    if url_col is not None:
        df = df.drop_duplicates(subset=[url_col])
    else:
        df = df.drop_duplicates()
    rows_after_duplicates = len(df)

    df_clean = pd.DataFrame()
    df_clean["city"] = df[city_col].astype(str)
    df_clean["city_label"] = df_clean["city"].map(CITY_NAME_MAP).fillna(df_clean["city"])

    if district_col is not None:
        df_clean["district"] = df[district_col].astype(str)
    else:
        df_clean["district"] = "Unknown"

    if address_col is not None:
        df_clean["address"] = df[address_col].astype(str)
    else:
        df_clean["address"] = "Unknown"

    df_clean["area_m2"] = df[area_col].apply(extract_first_number)
    df_clean["total_price_10k_yuan"] = df[total_price_col].apply(extract_first_number)
    df_clean["unit_price_yuan_m2"] = df[unit_price_col].apply(extract_first_number)

    if room_col is not None:
        room_counts = df[room_col].apply(extract_room_counts)
        df_clean["bedrooms"] = room_counts.apply(lambda x: x[0])
        df_clean["living_rooms"] = room_counts.apply(lambda x: x[1])
        df_clean["room_type_raw"] = df[room_col].astype(str)
    else:
        df_clean["bedrooms"] = np.nan
        df_clean["living_rooms"] = np.nan
        df_clean["room_type_raw"] = "Unknown"

    df_clean["room_type"] = df_clean.apply(
        lambda row: create_room_type_label(row["bedrooms"], row["living_rooms"]), axis=1
    )

    if floor_col is not None:
        df_clean["floor_raw"] = df[floor_col].astype(str)
    else:
        df_clean["floor_raw"] = "Unknown"

    if orientation_col is not None:
        df_clean["orientation_raw"] = df[orientation_col].astype(str)
        df_clean["orientation_group"] = df[orientation_col].apply(simplify_orientation)
    else:
        df_clean["orientation_raw"] = "Unknown"
        df_clean["orientation_group"] = "Other/Unknown"

    if year_col is not None:
        df_clean["building_year"] = df[year_col].apply(extract_year)
    else:
        df_clean["building_year"] = np.nan

    df_clean["building_age"] = CURRENT_YEAR - df_clean["building_year"]
    df_clean.loc[(df_clean["building_age"] < 0) | (df_clean["building_age"] > 120), "building_age"] = np.nan

    if url_col is not None:
        df_clean["origin_url"] = df[url_col].astype(str)
    else:
        df_clean["origin_url"] = "Unknown"

    df_clean["city_type"] = np.where(df_clean["city"].isin(MUNICIPALITIES), "Municipality", "Non-municipality")

    essential_numeric = ["area_m2", "total_price_10k_yuan", "unit_price_yuan_m2"]
    df_clean = df_clean.dropna(subset=essential_numeric)

    df_clean = df_clean[
        (df_clean["area_m2"].between(20, 500))
        & (df_clean["total_price_10k_yuan"].between(5, 1000))
        & (df_clean["unit_price_yuan_m2"].between(500, 50000))
    ]

    df_clean = df_clean[
        df_clean["bedrooms"].isna() | df_clean["bedrooms"].between(1, 10)
    ]
    df_clean = df_clean[
        df_clean["living_rooms"].isna() | df_clean["living_rooms"].between(0, 12)
    ]

    df_clean["price_segment"] = pd.qcut(
        df_clean["total_price_10k_yuan"],
        q=4,
        labels=["Low price", "Mid price", "High price", "Luxury"],
        duplicates="drop",
    )

    age_bins = [0, 5, 10, 20, 30, 50, 120]
    age_labels = ["0-5 years", "6-10 years", "11-20 years", "21-30 years", "31-50 years", "51-120 years"]
    df_clean["building_age_group"] = pd.cut(
        df_clean["building_age"].astype("float64"),
        bins=age_bins,
        labels=age_labels,
        include_lowest=True,
    )

    rows_after_cleaning = len(df_clean)

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
                len(df_raw),
                rows_after_essential,
                rows_after_duplicates,
                rows_after_cleaning,
                len(df_raw) - rows_after_cleaning,
            ],
        }
    )
    cleaning_summary["percentage_of_raw"] = (cleaning_summary["row_count"] / len(df_raw) * 100).round(2)
    cleaning_summary.to_csv(OUTPUT_DIR / "02_cleaning_summary.csv", index=False)

    df_clean.to_csv(OUTPUT_DIR / "03_cleaned_house_sales.csv", index=False, encoding="utf-8-sig")

    numeric_columns = [
        "area_m2",
        "total_price_10k_yuan",
        "unit_price_yuan_m2",
        "bedrooms",
        "living_rooms",
        "building_age",
    ]
    numeric_summary = df_clean[numeric_columns].describe().T.reset_index().rename(columns={"index": "variable"})
    numeric_summary.to_csv(OUTPUT_DIR / "04_numeric_summary.csv", index=False)

    correlation_matrix = df_clean[numeric_columns].corr(numeric_only=True)
    correlation_matrix.to_csv(OUTPUT_DIR / "05_correlation_matrix.csv")

    city_summary = (
        df_clean.groupby(["city", "city_label"])
        .agg(
            listing_count=("city", "size"),
            median_unit_price_yuan_m2=("unit_price_yuan_m2", "median"),
            mean_unit_price_yuan_m2=("unit_price_yuan_m2", "mean"),
            median_total_price_10k_yuan=("total_price_10k_yuan", "median"),
            median_area_m2=("area_m2", "median"),
        )
        .reset_index()
    )
    city_summary = city_summary[city_summary["listing_count"] >= 20]
    city_summary = city_summary.sort_values("median_unit_price_yuan_m2", ascending=False)
    city_summary.to_csv(OUTPUT_DIR / "06_city_price_summary.csv", index=False, encoding="utf-8-sig")

    price_segment_summary = (
        df_clean.groupby("price_segment", observed=False)
        .agg(
            listing_count=("total_price_10k_yuan", "size"),
            median_total_price_10k_yuan=("total_price_10k_yuan", "median"),
            median_unit_price_yuan_m2=("unit_price_yuan_m2", "median"),
            median_area_m2=("area_m2", "median"),
            median_building_age=("building_age", "median"),
        )
        .reset_index()
    )
    price_segment_summary.to_csv(OUTPUT_DIR / "07_price_segment_summary.csv", index=False)

    room_type_counts = df_clean["room_type"].value_counts().reset_index()
    room_type_counts.columns = ["room_type", "listing_count"]
    room_type_counts.to_csv(OUTPUT_DIR / "08_room_type_counts.csv", index=False)

    room_type_price_summary = (
        df_clean.groupby("room_type")
        .agg(
            listing_count=("room_type", "size"),
            median_unit_price_yuan_m2=("unit_price_yuan_m2", "median"),
            median_total_price_10k_yuan=("total_price_10k_yuan", "median"),
            median_area_m2=("area_m2", "median"),
        )
        .reset_index()
        .sort_values("median_unit_price_yuan_m2", ascending=False)
    )
    room_type_price_summary.to_csv(OUTPUT_DIR / "09_room_type_price_summary.csv", index=False)

    orientation_summary = (
        df_clean.groupby("orientation_group")
        .agg(
            listing_count=("orientation_group", "size"),
            median_unit_price_yuan_m2=("unit_price_yuan_m2", "median"),
            median_total_price_10k_yuan=("total_price_10k_yuan", "median"),
        )
        .reset_index()
        .sort_values("median_unit_price_yuan_m2", ascending=False)
    )
    orientation_summary.to_csv(OUTPUT_DIR / "10_orientation_price_summary.csv", index=False)

    building_age_summary = (
        df_clean.dropna(subset=["building_age_group"])
        .groupby("building_age_group", observed=False)
        .agg(
            listing_count=("building_age_group", "size"),
            median_unit_price_yuan_m2=("unit_price_yuan_m2", "median"),
            median_total_price_10k_yuan=("total_price_10k_yuan", "median"),
            median_building_age=("building_age", "median"),
        )
        .reset_index()
    )
    building_age_summary.to_csv(OUTPUT_DIR / "11_building_age_price_summary.csv", index=False)

    city_type_summary = (
        df_clean.groupby("city_type")
        .agg(
            listing_count=("city_type", "size"),
            median_unit_price_yuan_m2=("unit_price_yuan_m2", "median"),
            median_total_price_10k_yuan=("total_price_10k_yuan", "median"),
            median_area_m2=("area_m2", "median"),
        )
        .reset_index()
    )
    city_type_summary.to_csv(OUTPUT_DIR / "12_city_type_summary.csv", index=False)

    print("\nCleaning summary:")
    print(cleaning_summary)

    print("\nCleaned dataset shape:")
    print(df_clean.shape)

    print("\nNumeric summary:")
    print(numeric_summary)

    print("\nTop 10 cities by median unit price:")
    print(city_summary.head(10))

    # Chart 1: correlation matrix
    plt.figure(figsize=(8, 6))
    matrix = correlation_matrix.values
    plt.imshow(matrix, aspect="auto")
    plt.colorbar(label="Correlation")
    plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=45, ha="right")
    plt.yticks(range(len(correlation_matrix.index)), correlation_matrix.index)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if not np.isnan(value):
                plt.text(j, i, f"{value:.2f}", ha="center", va="center")
    plt.title("Correlation Matrix of Key Housing Variables")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "a1_correlation_matrix.png", dpi=300)
    plt.close()

    # Chart 2: total price distribution
    plt.figure(figsize=(10, 6))
    plt.hist(df_clean["total_price_10k_yuan"], bins=50)
    plt.title("Distribution of Total House Price")
    plt.xlabel("Total price (10,000 yuan)")
    plt.ylabel("Number of listings")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "a2_total_price_distribution.png", dpi=300)
    plt.close()

    # Chart 3: unit price boxplot
    plt.figure(figsize=(7, 5))
    plt.boxplot(df_clean["unit_price_yuan_m2"].dropna())
    plt.title("Boxplot of Unit Price")
    plt.ylabel("Unit price (yuan/m²)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "a2_unit_price_boxplot.png", dpi=300)
    plt.close()

    # Chart 4: top cities by unit price
    top_cities = city_summary.head(10).set_index("city_label")["median_unit_price_yuan_m2"]
    save_bar_chart(
        top_cities,
        "Top 10 Cities by Median Unit Price",
        "City",
        "Median unit price (yuan/m²)",
        "a3_top_city_unit_prices.png",
        figsize=(10, 6),
    )

    # Chart 5: price segment characteristics
    area_by_segment = price_segment_summary.set_index("price_segment")["median_area_m2"]
    save_bar_chart(
        area_by_segment,
        "Median Area by Price Segment",
        "Price segment",
        "Median area (m²)",
        "a4_median_area_by_price_segment.png",
        figsize=(9, 5),
    )

    age_by_segment = price_segment_summary.set_index("price_segment")["median_building_age"]
    save_bar_chart(
        age_by_segment,
        "Median Building Age by Price Segment",
        "Price segment",
        "Median building age (years)",
        "a4_building_age_by_price_segment.png",
        figsize=(9, 5),
    )

    # Chart 6: room type analysis
    top_room_counts = room_type_counts.head(10).set_index("room_type")["listing_count"]
    save_bar_chart(
        top_room_counts,
        "Top 10 Room Types by Listing Count",
        "Room type",
        "Number of listings",
        "a5_top_room_types.png",
        figsize=(11, 6),
    )

    common_room_types = room_type_price_summary[room_type_price_summary["listing_count"] >= 50].head(10)
    room_price_series = common_room_types.set_index("room_type")["median_unit_price_yuan_m2"]
    save_bar_chart(
        room_price_series,
        "Median Unit Price by Common Room Type",
        "Room type",
        "Median unit price (yuan/m²)",
        "a5_room_type_unit_price.png",
        figsize=(11, 6),
    )

    # Chart 7: orientation and age analysis
    orientation_series = orientation_summary.set_index("orientation_group")["median_unit_price_yuan_m2"]
    save_bar_chart(
        orientation_series,
        "Median Unit Price by Home Orientation",
        "Orientation",
        "Median unit price (yuan/m²)",
        "a6_orientation_unit_price.png",
        figsize=(9, 5),
    )

    age_series = building_age_summary.set_index("building_age_group")["median_unit_price_yuan_m2"]
    save_bar_chart(
        age_series,
        "Median Unit Price by Building Age Group",
        "Building age group",
        "Median unit price (yuan/m²)",
        "a7_building_age_unit_price.png",
        figsize=(10, 5),
    )

    # Chart 8: municipality comparison
    city_type_series = city_type_summary.set_index("city_type")["median_unit_price_yuan_m2"]
    save_bar_chart(
        city_type_series,
        "Median Unit Price: Municipality vs Non-municipality",
        "City type",
        "Median unit price (yuan/m²)",
        "a8_municipality_vs_non_municipality.png",
        figsize=(7, 5),
    )

    with open(OUTPUT_DIR / "13_project_summary.txt", "w", encoding="utf-8") as f:
        f.write("Real Estate Market Analysis and Value Evaluation\n")
        f.write("================================================\n\n")
        f.write(f"Raw dataset shape: {df_raw.shape}\n")
        f.write(f"Cleaned dataset shape: {df_clean.shape}\n")
        f.write("\nCleaning summary:\n")
        f.write(cleaning_summary.to_string(index=False))
        f.write("\n\nMain findings:\n")
        f.write("1. The total price distribution is right-skewed, meaning most listings are in lower and middle price ranges while fewer listings are very expensive.\n")
        f.write("2. Total price is strongly related to unit price and moderately related to property area.\n")
        f.write("3. City-level comparison shows clear differences in median unit price across cities.\n")
        f.write("4. Price segment analysis shows that higher price groups do not only depend on area; location and unit price also influence value.\n")
        f.write("5. Room type, orientation, building age, and city type provide useful ways to compare housing value patterns.\n")
        f.write("\nProfessional reflection:\n")
        f.write("This project helped me practise data cleaning, feature extraction, descriptive analysis, grouped comparison, visualisation, and project documentation.\n")

    print("\nProject completed successfully.")
    print(f"All output files have been saved in: {OUTPUT_DIR}")

    print("\nGenerated key files:")
    for name in [
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
        "a3_top_city_unit_prices.png",
        "a5_top_room_types.png",
        "a8_municipality_vs_non_municipality.png",
    ]:
        print(f"- {name}")


if __name__ == "__main__":
    main()
