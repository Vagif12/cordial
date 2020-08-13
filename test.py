from content_recommenders import GraphRecommender,BasicRecommender

recommender = GraphRecommender('/Users/Vagif/Downloads/disney_plus_shows.csv',feature_names=['genre','actors','writer','director'])
print(recommender.recommend('Toy Story'))