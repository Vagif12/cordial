import unittest
from cordial.content_recommenders import GraphRecommender, BasicRecommender,LDARecommender,LDADistanceRecommender

class TestQueries(unittest.TestCase):
    """
    Test recommender system queries for Travis
    """

    def test_graph_recommender(self):
        recommender = GraphRecommender('disney',text_feature='plot')
        assert 'result' in recommender.recommend('Coco')

    def test_basic_recommender(self):
        recommender = BasicRecommender('disney')
        assert 'result' in recommender.recommend('Toy Story')

    def test_lda_recommender(self):
        recommender = LDARecommender('disney')
        assert 'result' in recommender.recommend('Onward')

    def test_lda_distance_recommender(self):
        recommender = LDADistanceRecommender('disney')
        assert 'result' in recommender.recommend('Toy Story 2')

if __name__ == '__main__':
    unittest.main()
