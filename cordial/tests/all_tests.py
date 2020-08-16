import unittest
from cordial.content_recommenders import GraphRecommender, BasicRecommender,LDARecommender

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

    def test_basic_recommender(self):
        recommender = LDARecommender('disney')
        assert 'result' in recommender.recommend('Toy Story 4')

if __name__ == '__main__':
    unittest.main()
