from content_recommenders import GraphRecommender,BasicRecommender

recommender = GraphRecommender('disney')
print(recommender.recommend('Toy Story'))