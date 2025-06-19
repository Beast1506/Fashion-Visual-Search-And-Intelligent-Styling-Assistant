def recommend_outfit(query_category, jeans_df, dresses_df, top_k=3):
    if query_category == 'dress':
        # Recommend jeans
        return jeans_df.sample(top_k)
    else:
        # Recommend dresses
        return dresses_df.sample(top_k)