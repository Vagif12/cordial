import unittest
from cordial.content_recommenders import GraphRecommender, BasicRecommender

class TestQueries(unittest.TestCase):
    """
    Test recommender system queries for Travis CI
    """

    def test_graph_recommender(self):
        recommender = GraphRecommender('disney',text_feature='plot')
        assert 'result' in recommender.recommend('Coco')

    def test_basic_recommender(self):
        recommender = BasicRecommender('disney',text_feature='plot')
        assert 'result' in recommender.recommend('Toy Story')

if __name__ == '__main__':
    unittest.main()
