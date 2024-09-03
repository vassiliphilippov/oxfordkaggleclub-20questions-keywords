from pathlib import Path
from typing import Optional
import json
from typing import Any
import pandas as pd
import llm_easy_toolkit as let


def prepare_keywords_for_category(category: str, llm) -> pd.DataFrame:
    df_new_keywords: pd.DataFrame = pd.DataFrame(columns=["category", "keyword"])
    
    for _ in range(5):  # Run the generation 3 times sequentially
        new_keywords: str = let.run(
            llm=llm,
            template_path="prompts/keyword_generation_no_examples.txt",
            category=category,
            number=100,
        )
        
        new_keywords_dicts = [
            {"category": category, "keyword": keyword.strip()}
            for keyword in new_keywords.split("\n")
            if keyword.strip()
        ]

        df_new_keywords = pd.concat([df_new_keywords, pd.DataFrame(new_keywords_dicts)], ignore_index=True)
    
    return df_new_keywords


def main() -> None:
    folder_path: Path = Path("categories")
    keywords_folder_path: Path = Path("keywords")

    # load the list of categories from categories.csv using pandas
    categories: pd.DataFrame = pd.read_csv(folder_path / "categories.csv")

    llm: let.LLamaLLM = let.LLamaLLM(
        model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
    )

    # create a new DataFrame to store the new keywords
    df_new_keywords: pd.DataFrame = pd.DataFrame(columns=["category", "keyword"])

    # enumerate all categories
    num_of_categories: int = len(categories)
    for i, raw in enumerate(categories["category"]):
        category: str = raw
        try:
            category_keywords: pd.DataFrame = prepare_keywords_for_category(category, llm)
            df_new_keywords = pd.concat([df_new_keywords, category_keywords], ignore_index=True)
        except Exception as exc:
            print(f"Error for category {category}: {exc}")
        
        # save the updated new_keywords DataFrame to a CSV file after each category
        df_new_keywords.to_csv(keywords_folder_path / "category_keywords.csv", index=False)
        
        # Print progress
        print(f"Processed {i+1}/{num_of_categories} categories")


if __name__ == '__main__':
    main()
