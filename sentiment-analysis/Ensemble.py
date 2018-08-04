from .CorpusCreation import CorpusCreator
from .CorpusAnalysis import CorpusAnalyser
from .RedditScraper import ScrapeReddit
from .RedditSentiment import SentimentAnalyser

class Assembler:

    def __init__(self, directory):
        self.directory = directory
        self.scraper = ScrapeReddit(directory)
        self.sentiment = SentimentAnalyser(directory)
        self.corpus = CorpusCreator(directory)
        self.model = CorpusAnalyser(directory)

    def load(self, file_name):
        self.corpus.load_corpus(file_name)

    def prep(self, max_features):
        self.corpus.clean_data()
        self.corpus.tfidf_vectorize(max_features=max_features)
        self.corpus.normalize()

    def reduce_dimensions(self, method, data, parameter):
        self.model.load_data(data)
        if method.lower() == "t-sne":
            self.model.t_sne(perplexity=parameter)
        elif method.lower() == "pca":
            self.model.reduce_dimensions_pca(components=parameter)
        else:
           print("Reduction method not recognized")

    def cluster(self, method, clusters):
        if method.lower() == "kmeans":
            self.model.k_means(clusters=clusters)
        else:
            print("Clustering method not recognized")

    def score(self, method):
        if method.lower() == "silhouette":
            self.model.record_silhouette_score()
        else:
            print("Scoring method not recognized")

    def show(self):
        headlines = self.corpus.get_headlines()
        self.model.hover_plot(headlines)


s = Assembler(directory="/Users/cameronlaedtke/PycharmProjects/MLPractice/RedditNLP/data/")

s.load(file_name="the_donald/headlines.csv")
s.load(file_name="politics/headlines.csv")
s.load(file_name="The_Mueller/headlines.csv")

s.prep(max_features=2000)
s.reduce_dimensions(method="pca", data=s.corpus, parameter=30)
s.reduce_dimensions(method="t-sne", data=s.corpus, parameter=36)
s.cluster(method="kmeans", clusters=12)
s.score(method="silhouette")
s.show()

