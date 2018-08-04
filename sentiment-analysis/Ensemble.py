from .CorpusCreation import CorpusCreator
from .CorpusAnalysis import CorpusAnalyser
from .RedditScraper import ScrapeReddit
from .RedditSentiment import SentimentAnalyser

sub = ScrapeReddit(directory="/Users/cameronlaedtke/PycharmProjects/MLPractice/RedditNLP/data/")

sub.set_subreddit("funny")
sub.get_posts()
sub.get_comments()

sent = SentimentAnalyser(directory="/Users/cameronlaedtke/PycharmProjects/MLPractice/RedditNLP/data/",
                          subreddit="The_Mueller")

sent.run_sentiment_analysis()
sent.word_distribution('politics/negative_list.txt')
sent.run_topical_analysis("Kushner")
sent.stacked_bar(group="topics")
sent.clear_files()

class Assembler:

    def __init__(self, directory):
        self.directory = directory
        self.scraper = ScrapeReddit(directory)
        self.corpus = CorpusCreator(directory)
        self.model = CorpusAnalyser(directory)

    def load(self, file_name):
        self.corpus.load_corpus(file_name)

    def prep(self, max_features):
        self.corpus.clean_data()
        self.corpus.tfidf_vectorize(max_features=max_features)
        self.corpus.normalize()

    def reduce(self, method, data, parameter):
        self.model.load_array(data)
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
s.reduce(method="pca", data=corpus, parameter=30)
s.reduce(method="t-sne", data=corpus, parameter=36)
s.cluster(method="kmeans", clusters=12)
s.score(method="silhouette")
s.show()

