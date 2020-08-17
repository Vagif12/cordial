from cordial.content_recommenders import LDADistanceRecommender

recommender = LDADistanceRecommender('disney',feature_names=['genre','plot','writer','actors','director'])
print(recommender.recommend('Toy Story')['result'])