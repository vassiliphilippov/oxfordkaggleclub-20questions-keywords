## Keywords

Our solution works with a pre-calculated list of potential keywords.
There is a dilemma of whether to use a large keyword list to maximize coverage, which will require more questions to guess, 
or use a smaller list of keywords that are likely included in the private keyword list, which offers smaller coverage but requires fewer questions to guess.
Initially, we considered choosing one approach based on the expected meta game, but then we discovered that it is possible to combine these two approaches 
by adding a probability for each keyword to be in the private keyword list. So our keyword list is a dataframe with two columns: keyword and probability.

To generate the list of keywords with probabilities, we did the following:

1. Split the public keyword list 50/50 into training and validation sets
2. Categorize all the training keywords into 12 categories (e.g., Home and Living, Technology and Electronics, Hand Tools, etc.)
3. Split each category into several subcategories (e.g., Home and Living -> Furniture, Appliances, Kitchen Items, Home Decor, etc.)
4. Further divide each subcategory into third-level subsubcategories (e.g., Furniture -> Seating, Tables, Storage, Beds, Outdoor Furniture), resulting in a total of ~1700 third-level subsubcategories
5. Use an LLM to generate 100 possible keywords for each subsubcategory, using relevant training keywords as examples
6. Collect all the generated keywords into one large CSV file
7. Repeat steps #5-#6 five times
8. Count how many times each keyword was generated, assuming that a higher count indicates a higher probability of the keyword appearing in the private keyword list
9. Add the most popular English nouns with counts depending on their frequency
10. Add the list of countries and cities with low probability just in case