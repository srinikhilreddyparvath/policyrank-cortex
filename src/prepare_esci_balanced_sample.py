import os
import pandas as pd


EXAMPLES_PATH = (
    "external_data/esci-data/shopping_queries_dataset/"
    "shopping_queries_dataset_examples.parquet"
)

PRODUCTS_PATH = (
    "external_data/esci-data/shopping_queries_dataset/"
    "shopping_queries_dataset_products.parquet"
)

OUTPUT_PATH = "data/esci_balanced_sample.csv"

TARGET_ROWS = 50000
MAX_ROWS_PER_QUERY = 30
RANDOM_SEED = 42


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.lower().strip() for c in df.columns]
    return df


def find_column(df: pd.DataFrame, possible_names):
    for name in possible_names:
        if name in df.columns:
            return name
    return None


def create_query_balanced_sample(examples: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a query-balanced sample without using groupby.apply.

    Logic:
    1. Shuffle all rows.
    2. Within each query, assign row number.
    3. Keep up to MAX_ROWS_PER_QUERY per query.
    4. Sample TARGET_ROWS from the balanced pool.
    """

    shuffled = examples.sample(
        frac=1.0,
        random_state=RANDOM_SEED,
    ).reset_index(drop=True)

    shuffled["row_num_within_query"] = shuffled.groupby("query").cumcount()

    balanced_pool = shuffled[
        shuffled["row_num_within_query"] < MAX_ROWS_PER_QUERY
    ].copy()

    balanced_pool = balanced_pool.drop(columns=["row_num_within_query"])

    if len(balanced_pool) > TARGET_ROWS:
        balanced = balanced_pool.sample(
            n=TARGET_ROWS,
            random_state=RANDOM_SEED,
        ).reset_index(drop=True)
    else:
        balanced = balanced_pool.reset_index(drop=True)

    return balanced


def main():
    print("Loading ESCI examples...")

    examples = pd.read_parquet(EXAMPLES_PATH)
    examples = normalize_column_names(examples)

    print("Examples columns:")
    print(examples.columns.tolist())

    query_col = find_column(examples, ["query", "query_text", "search_query"])
    product_id_col = find_column(examples, ["product_id", "asin", "product_asin"])
    esci_col = find_column(examples, ["esci_label", "esci", "label"])

    if query_col is None:
        raise ValueError("Could not find query column in examples parquet.")

    if product_id_col is None:
        raise ValueError("Could not find product_id/asin column in examples parquet.")

    if esci_col is None:
        raise ValueError("Could not find ESCI label column in examples parquet.")

    examples = examples[[query_col, product_id_col, esci_col]].copy()

    examples = examples.rename(
        columns={
            query_col: "query",
            product_id_col: "product_id",
            esci_col: "esci_label",
        }
    )

    examples["query"] = examples["query"].astype(str).str.strip()
    examples["product_id"] = examples["product_id"].astype(str).str.strip()
    examples["esci_label"] = examples["esci_label"].astype(str).str.strip()

    examples = examples.dropna(subset=["query", "product_id", "esci_label"])
    examples = examples[examples["query"].str.len() > 0]
    examples = examples[examples["product_id"].str.len() > 0]

    print("Total example rows:", len(examples))
    print("Unique queries:", examples["query"].nunique())

    print("Creating query-balanced sample...")

    balanced = create_query_balanced_sample(examples)

    print("Balanced rows before product join:", len(balanced))
    print("Balanced unique queries:", balanced["query"].nunique())

    print("Loading product metadata...")

    products = pd.read_parquet(PRODUCTS_PATH)
    products = normalize_column_names(products)

    print("Products columns:")
    print(products.columns.tolist())

    prod_product_id_col = find_column(products, ["product_id", "asin", "product_asin"])
    title_col = find_column(products, ["product_title", "title", "product_name"])
    desc_col = find_column(products, ["product_description", "description", "product_desc"])
    category_col = find_column(products, ["product_category", "category", "product_type"])
    brand_col = find_column(products, ["product_brand", "brand"])

    if prod_product_id_col is None:
        raise ValueError("Could not find product_id/asin column in products parquet.")

    keep_cols = [prod_product_id_col]

    if title_col:
        keep_cols.append(title_col)

    if desc_col:
        keep_cols.append(desc_col)

    if category_col:
        keep_cols.append(category_col)

    if brand_col:
        keep_cols.append(brand_col)

    products = products[keep_cols].copy()

    rename_map = {prod_product_id_col: "product_id"}

    if title_col:
        rename_map[title_col] = "product_title"

    if desc_col:
        rename_map[desc_col] = "product_description"

    if category_col:
        rename_map[category_col] = "category"

    if brand_col:
        rename_map[brand_col] = "brand"

    products = products.rename(columns=rename_map)
    products["product_id"] = products["product_id"].astype(str).str.strip()

    print("Joining examples with product metadata...")

    merged = balanced.merge(
        products,
        on="product_id",
        how="left",
    )

    if "product_title" not in merged.columns:
        merged["product_title"] = ""

    if "product_description" not in merged.columns:
        merged["product_description"] = ""

    if "category" not in merged.columns:
        merged["category"] = "unknown"

    if "brand" not in merged.columns:
        merged["brand"] = "unknown"

    merged["product_title"] = merged["product_title"].fillna("").astype(str)
    merged["product_description"] = merged["product_description"].fillna("").astype(str)
    merged["category"] = merged["category"].fillna("unknown").astype(str)
    merged["brand"] = merged["brand"].fillna("unknown").astype(str)

    # Keep synthetic price/rating for now because ESCI product metadata does not
    # reliably provide normalized retail price/rating fields for our app.
    merged["price"] = 50.0
    merged["rating"] = 4.0

    merged["search_text"] = (
        merged["product_title"].astype(str)
        + " "
        + merged["product_description"].astype(str)
        + " "
        + merged["category"].astype(str)
        + " "
        + merged["brand"].astype(str)
    )

    output_cols = [
        "product_id",
        "product_title",
        "product_description",
        "category",
        "brand",
        "query",
        "esci_label",
        "price",
        "rating",
        "search_text",
    ]

    merged = merged[output_cols].copy()

    os.makedirs("data", exist_ok=True)
    merged.to_csv(OUTPUT_PATH, index=False)

    print("Done.")
    print("Saved to:", OUTPUT_PATH)
    print("Rows:", len(merged))
    print("Unique queries:", merged["query"].nunique())
    print("Unique products:", merged["product_id"].nunique())

    print("Label distribution:")
    print(merged["esci_label"].value_counts(dropna=False))

    print("Sample queries:")
    print(merged["query"].drop_duplicates().head(30).to_string(index=False))

    print("Sample rows:")
    print(
        merged[
            [
                "query",
                "product_title",
                "category",
                "brand",
                "esci_label",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()