
import re
import pandas as pd

DATA_PATH = "data/questions (1).csv"
OUTPUT_PATH = "data/cleaned_questions.csv"


def clean_statement(text):
    text = str(text)

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove repeated dataset boilerplate
    text = re.sub(
        r"All submissions for this problem are available\.?",
        " ",
        text,
        flags=re.IGNORECASE
    )

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def main():
    df = pd.read_csv(DATA_PATH)

    # Keep records with the fields we need
    df = df.dropna(subset=["QCode", "Title", "statement"]).copy()

    for column in ["QCode", "Title", "statement"]:
        df[column] = df[column].astype(str).str.strip()

    df = df[
        (df["QCode"] != "") &
        (df["Title"] != "") &
        (df["statement"] != "")
    ]

    df["clean_statement"] = df["statement"].apply(clean_statement)

    # Remove empty cleaned statements and duplicate text
    df = df[df["clean_statement"].str.len() > 0]
    df = df.drop_duplicates(subset=["clean_statement"])
    df = df.reset_index(drop=True)

    df.to_csv(OUTPUT_PATH, index=False)

    print("Data cleaning completed!")
    print("Original file rows:", len(pd.read_csv(DATA_PATH)))
    print("Cleaned problem count:", len(df))
    print("Saved to:", OUTPUT_PATH)

    print("\nExample:")
    print("Title:", df.iloc[0]["Title"])
    print("Cleaned statement:", df.iloc[0]["clean_statement"][:300])


if __name__ == "__main__":
    main()
