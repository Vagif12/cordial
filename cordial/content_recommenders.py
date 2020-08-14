import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity,linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
import networkx as nx
import math


# Cleaning/ preprocessing function for graph based recommender engine
# basically converts feature values to list and fills in any NaN values
def preprocess(df,feature_names,indexer):


    # check if feature names were passed in, if not automaticallt detect
    if feature_names == [] or feature_names == None:
        # Getting defaults if no feature names were passed
        options = ['id','_id','rating','_rating','score','_score','rated']
        feature_names = list(df.select_dtypes('object').columns)
        for i in options:
            if i in [x.lower() for x in feature_names]:
                del feature_names[feature_names.index(i)]

    # convert elements to list and fill NaN values
    for f in feature_names:
        if f.lower() not in ['description','overview','plot','summary','text',indexer]:
            df = df.copy()
            df[f] = df[f].apply(lambda l: [] if pd.isna(l) else [i.strip() for i in l.split(",")])
            df = df.fillna('')
    return df,feature_names



class GraphRecommender:
    """ Main class for the Graph Recommender """
    """
    Concretely, this returns the top n recommendations for the specified value based on given features.
    Parameters:

    dataset: dataset name
    feature_names: a list of features to be used to generate recommendations
    search_name: the name of the item you are looking to get recommendations from
    n_recommendations: number of recommendations(default=10)
    indexer: the name of the feature you want to get recommendations from
    text_feature: the feature that describes the item your are trying to recommend(description,plot,overview,etc)

    Example:

    For a movie dataset with features incliding genre,actors,writers:

    c.content_recommender_query(feature_names=['genre','actors','writer','director','country'],text_feature='plot')
    recommendations = c.recommend('Toy Story')

    Returns:
    a dictionary with the following keys:

    recommendations: the generated recommendations
    n_recommendations: the number of recommendations made
    indexer: the indexer used
    text_feature: the text feature used
    feature_names: the feature names used

    """
    def __init__(self,dataset,feature_names=[],text_feature=None,indexer='title',n_recommendations=10):
        if not dataset.endswith('.csv'):
            ds = 'https://raw.githubusercontent.com/Vagif12/cordial/master/datasets/{}.csv'.format(dataset)
            dataset = ds
        self.df,self.feature_names = preprocess(pd.read_csv(dataset),feature_names,indexer)
        print('|- Preprocessing data... -|')
        # check if txt feature was passed in, if not then automatically detect
        self.text_feature = text_feature
        if text_feature == '' or text_feature == None:
            try:
                opt = ['description','overview','plot','summary','text']
                for i in opt:
                    if i in list(self.df.columns):
                        self.text_feature = i
                        break
                else:
                    self.text_feature = text_feature
            except:
                print('|- Text Feature was not detected. Please provide one! -|')

        # initialisations
        self.G = nx.Graph()
        self.indexer = indexer
        self.n_recommendations = n_recommendations

        # check if feature names is a list, if not throw error
        if not(isinstance(self.feature_names,list)):
            print('|- Error: feature names must be a list!')
            exit()
        
    # main function
    def recommend(self,root):
        # TFIDF
        text_content = self.df[self.text_feature]
        vector = TfidfVectorizer(max_df=0.4,         
                                     min_df=1,      
                                     stop_words='english', 
                                     lowercase=True, 
                                     use_idf=True,   
                                     norm=u'l2',     
                                     smooth_idf=True 
                                    )
        tfidf = vector.fit_transform(text_content)

        # Mini-batch kmeans on tfidf vectorized text content
        k = 200
        kmeans = MiniBatchKMeans(n_clusters = k)
        kmeans.fit(tfidf)
        centers = kmeans.cluster_centers_.argsort()[:,::-1]
        terms = vector.get_feature_names()
        print('|- Getting Bag of Words.. -|')

        request_transform = vector.transform(text_content)
        self.df['cluster'] = kmeans.predict(request_transform) 

        # Find similiar indices        
        def find_similar(tfidf_matrix, index, top_n = 5):
            cosine_similarities = linear_kernel(tfidf_matrix[index:index+1], tfidf_matrix).flatten()
            related_docs_indices = [i for i in cosine_similarities.argsort()[::-1] if i != index]
            return [index for index in related_docs_indices][0:top_n] 

        print('|- Getting cosine similarities.. -|')

        # Add edges and nodes for each of the feature names
        for i, rowi in self.df.iterrows():
            self.G.add_node(rowi[self.indexer],label="ITEM")
            for f in self.feature_names:
                for element in rowi[f]:
                    self.G.add_node(element)
                    self.G.add_edge(rowi[self.indexer], element)

            indices = find_similar(tfidf, i, top_n = 5)
            snode="Sim("+rowi[self.indexer][:15].strip()+")"        
            self.G.add_node(snode,label="SIMILAR")
            self.G.add_edge(rowi[self.indexer], snode, label="SIMILARITY")
            for element in indices:
                self.G.add_edge(snode, self.df[self.indexer].loc[element], label="SIMILARITY")

        # The actual recommendation code
        commons_dict = {}
        print('|- Getting Recommendations.. -|')
        for e in self.G.neighbors(root):
            for e2 in self.G.neighbors(e):
                if e2==root:
                    continue
                if self.G.nodes[e2]['label']=="ITEM":
                    commons = commons_dict.get(e2)
                    if commons==None:
                        commons_dict.update({e2 : [e]})
                    else:
                        commons.append(e)
                        commons_dict.update({e2 : commons})
        items=[]
        weight=[]
        for key, values in commons_dict.items():
            w=0.0
            for e in values:
                w=w+1/math.log(self.G.degree(e))
            items.append(key) 
            weight.append(w)

        # Results are a DataFrame under the result key
        result = pd.DataFrame(data=np.array(weight),index=items).reset_index().sort_values(0,ascending=False)
        result.columns = ['Title','Similarity'] 
        output = result[:self.n_recommendations]  
        print("|- Done! Recommendations can be viewed as a DataFrame under the 'result' key! -|") 
        return {
            'result': output,
            'n_recommendations': self.n_recommendations,
            'indexer': self.indexer,
            'feature_names': self.feature_names,
            'text_feature': self.text_feature
        }


####################################### END OF GRAPH RECOMMENDER ##############################################################


# Cleaning/ preprocessing function for basic content based recommender engine
# basically tokenizes all the feature examples
def clean_data(x):
    if isinstance(x, list):
        return np.array([str.lower(i.replace(" ", "")) for i in x if not x.isdigit()])
    else:
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''


# Cosine similarity matrix creator
def matrix_maker(data,indexer='title',feature_names=[]):
    # check if feature names are of type list
    try:
        assert(isinstance(feature_names, list))
    except:
        print('|- Error! Feature names must be of type list! -|')
        exit()


    # function to combine all the tokenized features values for the
    # cosine matrix to calculate similarities from (the recommender 'soup')
    def create_soup(x):
        soup = []
        for feature in np.array(feature_names):
            f = ''.join(x[feature])
            soup.append(f)
        return ' '.join(soup)

    data = pd.read_csv(data)
    data = data.copy()
    try:
        assert indexer in data.columns
    except:
        print(' |- Error! the indexer passed named:  ' +  str(indexer) + 'is not in dataset! -|')
        exit()
    for f in feature_names:
            data[f] = data[f].apply(clean_data)
    print('|- Cleaning Data.. -|')

    data['text'] = data.apply(create_soup,axis=1)
    print('|- Getting similarities -|')

    # Create a CountVectorizer, fit to data 'soup' and get similarities
    con = vector = TfidfVectorizer(max_df=0.4,         
                                     min_df=1,      
                                     stop_words='english', 
                                     lowercase=True, 
                                     use_idf=True,   
                                     norm=u'l2',     
                                     smooth_idf=True 
                                    )
    item_matrix = con.fit_transform(data['text'])
    cosine_similarities = cosine_similarity(item_matrix,item_matrix)
    similarities = {}

    # Loop through similarities and get top 50, return similarities
    for i in range(len(cosine_similarities)):
        similar_indices = cosine_similarities[i].argsort()[:-50:-1] 
        similarities[data[indexer].iloc[i]] = [(cosine_similarities[i][x], data[indexer][x], '') for x in similar_indices][1:]
    return similarities

class BasicRecommender:
    '''
    ---------- Basic Content Recommender Class --------
    This is the base class for the basic content recommender. The constructor takes in 5 arguements:
    data: the dataset to work on
    feature_names = a list of the names of features you would like to use to get recommendations
    by default this will be all the categorical columns
    n_recommendations: the number of recommendations to return
    indexer = the name of the columns you want to get recommendations from
    by default this will be the first categorical columns(excluding the id)
    Example:
    newClient.content_recommender_query(feature_names=['genre','actors','writer','plot'],indexer='title')
    Methods:
    recommend: the recommendations function. Returns recommendations based on search term passed
    parameters:
    search_term: string of the item you want to get recommendations from
    returns:
    result: a pandas DataFrame of the top n recommendations
    Example:
    c = client('path to file')
    c.content_recommender_query(feature_names=['genre','plot','director','actors','writer'])
    recommendations = c.recommend('Coco')
    _get_message: gets the results of the similarity and creates a Dataframe
    of the resultd and their correlation.
    '''


    def __init__(self, dataset,feature_names=[],indexer='',n_recommendations=10):
        # If feature names is blank, then it get all categorical objects,
        # removes the id and used them to recommend items,setting the indexer
        # as the first element of the feature_names
        if not dataset.endswith('.csv'):
            ds = 'https://raw.githubusercontent.com/Vagif12/cordial/master/datasets/{}.csv'.format(dataset)
            dataset = ds

        self.data1 = pd.read_csv(dataset).copy()

        if feature_names == []:
            catnames = list(self.data1.select_dtypes('object').columns)
            for i in catnames:
                if 'id' in i.lower() or 'rating' in i.lower():
                    v = catnames.index(i)
                    del catnames[v]
            self.feature_names = catnames[1:]
            self.indexer = catnames[0]
        else:
            # Initialise default variables
            self.indexer = indexer
            self.feature_names = feature_names

        # call the matrix_maker function created above
        self.matrix_similar = matrix_maker(dataset,self.indexer,self.feature_names)
        self.n_recommendations=n_recommendations

    def _get_message(self, item, recom_items):
        # Get the recommendations and put them into a DataFrame
        print("|- Complete! Stored recommendation DataFrame under the 'recommendations' key! -|")

        rec_items = len(recom_items)
        # List for recommended items and their respective correlations
        recommended_items = []
        recommended_corr = []

        # Loop through each item,append to the correct list, and returnt a dict key
        # of the DataFrame,n_recommendations, feature_names and indexer
        for i in range(rec_items):
            recommended_items.append(recom_items[i][1])
            recommended_corr.append(round(recom_items[i][0], 3))

        
        df = pd.DataFrame(pd.Series(np.array(recommended_items)),columns=['Recommendations'])
        df.insert(1,'Correlation',recommended_corr)


        return {
            'result': df,
            'n_recommendations': self.n_recommendations,
            'indexer': self.indexer,
            'feature_names': self.feature_names,
        }

        
    # recommendation function
    def recommend(self, s_name):
        # Get item to find recommendations for
        item = s_name
        # Get number of items to recommend
        number_items = self.n_recommendations
        # Get the number of items most similars from matrix similarities
        recom_item = self.matrix_similar[item][:number_items]
        # return each item

        return self._get_message(item=item, recom_items=recom_item)