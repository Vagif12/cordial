from cordial.content_recommenders import LDARecommender

recommender = LDARecommender('disney')
print(recommender.recommend('Toy Story')['result'])