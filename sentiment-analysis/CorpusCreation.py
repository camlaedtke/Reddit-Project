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
        self.file_list = []
        print("initialized")

    def get_original_indices(self):
        return self.indices

    def get_headlines(self):
        return self.text_corpus

    def load_corpus(self, file_name):
        self.file_list = self.file_list.append(file_name)
        file_path = self.directory + file_name
        if len(self.file_list) == 0:
            self.raw_corpus = pd.read_csv(filepath_or_buffer=file_path, sep=",",
                                          names=["title", "author", "created_utc", "score",
                                          "link_flair_text", "domain","self_text", "id"])
        else:
            df = pd.read_csv(filepath_or_buffer=file_path, sep=",",
                             names=["title", "author", "created_utc", "score",
                                    "link_flair_text", "domain", "self_text", "id"])
            self.raw_corpus = pd.concat([self.raw_corpus, df])
        print("Data loaded")

    def chop_corpus(self, num_rows):
        self.raw_corpus = self.raw_corpus.sample(n=num_rows)
        self.indices = self.raw_corpus.index
        self.raw_corpus = self.raw_corpus.reset_index(drop=True)

    def clean_data(self):
        '''Create new column called TITLE_lower of titles but in lowercase, delete rows with blank titles'''
        self.raw_corpus = self.raw_corpus.reset_index(drop=True)
        self.raw_corpus = self.raw_corpus.loc[:, ["id", "author", "score", "title"]]
        self.text_corpus = self.raw_corpus.loc[:, "title"].str.lower()
        print('Data cleaned')

    def tfidf_vectorize(self, max_features):
        '''Tfidf vectorizer has lots of useful functions'''
        vectorizer = TfidfVectorizer(stop_words='english', max_features=max_features)
        self.vectorized = vectorizer.fit_transform(self.text_corpus)
        self.vectorized = self.vectorized.todense()
        print(self.vectorized.shape)
        print('Data vectorized')

    def normalize(self):
        '''Normalize data using standard scalar'''
        scaler = StandardScaler(copy=False)
        #self.X_normalized = scaler.partial_fit(self.X).transform(self.X)
        self.normalized = scaler.fit(self.vectorized).transform(self.vectorized)
        print("Data normalized")

    def numpy_normalize(self):
        '''By using numpy's implementation of std, memory consumption can be reduced by half'''
        std = self.X.std()
        mean = self.X.mean()
        scaler = StandardScaler(copy=False)
        scaler.std = std
        scaler.mean = mean
        self.X_normalized = scaler.fit_transform(self.X)
        print("Data normalized with numpy")

    def reduce_dimensions_PCA(self, components, svd_solver):
        '''Dimensionality reduction using principle component analysis (PCA), produces lower dimensional array'''
        pca = PCA(n_components=components, svd_solver=svd_solver)
        self.X_reduced = pca.fit_transform(self.normalized)

        print('Dimensions reduced to: '+str(pca.n_components_))

    def reduce_dimensions_UMAP(self):
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

