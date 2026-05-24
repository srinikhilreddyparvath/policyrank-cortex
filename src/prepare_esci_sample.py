import pandas as pd


EXAMPLES_PATH = "external_data/esci-data/shopping_queries_dataset/shopping_queries_dataset_examples.parquet"
PRODUCTS_PATH = "external_data/esci-data/shopping_queries_dataset/shopping_queries_dataset_products.parquet"
OUTPUT_PATH = "data/esci_sample_products.csv"


def main():
    print("Loading ESCI examples...")
    examples = pd.read_parquet(EXAMPLES_PATH)

    print("Loading ESCI products...")
    products = pd.read_parquet(PRODUCTS_PATH)

    print("Examples columns:")
    print(examples.columns.tolist())

    print("Products columns:")
    print(products.columns.tolist())

    print("Examples shape:", examples.shape)
    print("Products shape:", products.shape)

    # Keep English marketplace rows first for a smaller clean MVP.
    if "product_locale" in examples.columns:
        examples = examples[examples["product_locale"] == "us"]

    if "product_locale" in products.columns:
        products = products[products["product_locale"] == "us"]

    # Keep only a manageable sample.
    examples_sample = examples.head(5000)

    merged = examples_sample.merge(
        products,
        on=["product_locale", "product_id"],
        how="left"
    )

    # Build app-friendly product table.
    app_df = pd.DataFrame()

    app_df["product_id"] = merged["product_id"]
    app_df["product_title"] = merged["product_title"].fillna("")
    app_df["category"] = merged.get("product_type", "unknown")
    app_df["query"] = merged["query"]
    app_df["esci_label"] = merged["esci_label"]

    # Synthetic fields for current app compatibility.
    app_df["price"] = 50.0
    app_df["rating"] = 4.0

    app_df = app_df.dropna(subset=["product_id", "product_title", "query", "esci_label"])
    app_df = app_df.drop_duplicates(subset=["product_id", "query"])

    app_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved ESCI sample to: {OUTPUT_PATH}")
    print("Final shape:", app_df.shape)
    print(app_df.head(10))


if __name__ == "__main__":
    main()