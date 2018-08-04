import pandas as pd
import numpy as np
import operator
import pickle
import copy
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.exceptions import DataConversionWarning
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
#warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings("ignore", message=": RuntimeWarning: numpy.dtype size changed")
#warnings.filterwarnings("always")

class CorpusCreator():

    def __init__(self, directory):
        self.directory = directory
        print("initialized")

    def load_corpus(self, file_name):
        file_path = self.directory + file_name
        self.raw_corpus = pd.read_csv(filepath_or_buffer=file_path, sep=",",
                                      names=["title", "author", "created_utc", "score",
                                             "link_flair_text", "domain","self_text", "id"])

    def add_corpus(self, file_name):
        file_path = self.directory + file_name
        df = pd.read_csv(filepath_or_buffer=file_path, sep=",",
                                      names=["title", "author", "created_utc", "score",
                                             "link_flair_text", "domain", "self_text", "id"])
        #self.raw_corpus.append(df, ignore_index=True)
        self.raw_corpus = pd.concat([self.raw_corpus, df])
        print(len(self.raw_corpus))

    def cleanData(self):
        '''Create new column called TITLE_lower of titles but in lowercase, delete rows with blank titles'''
        self.raw_corpus = self.raw_corpus.loc[:, ["id", "author", "score", "title"]]
        self.text_corpus = self.raw_corpus.loc[:, "title"].str.lower()
        print('Data cleaned')

    def tfidfVectorize(self):
        '''Tfidf vectorizer has lots of useful functions'''
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        self.vectorized = vectorizer.fit_transform(self.text_corpus)
        print(self.vectorized.shape)
        self.vectorized = self.vectorized.todense()
        print(self.vectorized.shape)
        print('Data vectorized')

    def normalize(self):
        '''Normalize data using standard scalar'''
        scaler = StandardScaler(copy=False)
        #self.X_normalized = scaler.partial_fit(self.X).transform(self.X)
        self.normalized = scaler.fit(self.vectorized).transform(self.vectorized)
        print("Data normalized")

    def numpyNormalize(self):
        '''By using numpy's implementation of std, memory consumption can be reduced by half'''
        std = self.X.std()
        mean = self.X.mean()
        scaler = StandardScaler(copy=False)
        scaler.std = std
        scaler.mean = mean
        self.X_normalized = scaler.fit_transform(self.X)
        print("Data normalized with numpy")

    def reduceDimensionsPCA(self, components, svd_solver):
        '''Dimensionality reduction using principle component analysis (PCA), produces lower dimensional array'''
        pca = PCA(n_components=components, svd_solver=svd_solver)
        self.X_reduced = pca.fit_transform(self.normalized)

        print('Components: '+str(pca.n_components_))
        print('Dimensions reduced')

    def reduceDimensionsUMAP(self):
        print("hello")

    def clear_file(self, file_name):
        file_path = self.directory + file_name
        with open(file_path, "w+") as file:
            file.close()

    def create_file(self, name):
        '''Create file of array using pickle'''
        file_name = self.directory + "BagOfWords/" + name
        fileObject = open(file_name, 'wb')
        pickle.dump(self.normalized, fileObject)
        fileObject.close()
        print('File created')

    def get_array(self):
        return self.normalized

# corp = BagOfWords(directory="/Users/cameronlaedtke/PycharmProjects/MLPractice/RedditNLP/data/")
# #corp.clear_file("BagOfWords/headlines_array.csv")
# corp.load_corpus("the_donald/headlines.csv")
# #corp.add_corpus("The_Mueller/headlines.csv")
# #corp.add_corpus("politics/headlines.csv")
# corp.cleanData()
# corp.tfidfVectorize()
# corp.normalize()

#corp.create_file("headlines_array.csv")

# arrayObj = arrayCreation(size=50000)
# arrayObj.cleanData()
# arrayObj.tfidfVectorize()
# arrayObj.normalize()
# arrayObj.numpyNormalize()
# arrayObj.reduceDimensionsPCA(components=0.8, svd_solver='full')
# arrayObj.createFile(name='word_occurances_50k0.8C.csv')