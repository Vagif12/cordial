# cordial
Coridal is an intuitive light-weight API that allows developers to seamlessly use different recommender systems for their needs. 
In just three lines of a code, one can built a powerful content recommender system. Currently, Coridal only provides content recommenders,
but collaborative filtering is coming soon!

# Installation
`pip install cordial`

# Example usage:

```python

# An example with Cordial's BasicRecommender
from cordial.content_recommenders import GraphRecommender
recommender = BasicRecommender('disney')
print(recommender.recommend('Toy Story')['result'])

# An example with Cordial's GraphRecommender
from cordial.content_recommenders import GraphRecommender
recommender = GraphRecommender('netflix')
print(recommender.recommend('Toy Story')['result'])

# It's as simple as pie!
```

Full documentation can be viewed [here](https://github.com/Vagif12/cordial/wiki)
