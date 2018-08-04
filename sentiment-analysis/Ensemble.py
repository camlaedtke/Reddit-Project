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


def load(file_name):
    corpus.load_corpus(file_name)

def prep(max_features):
    corpus.clean_data()
    corpus.tfidf_vectorize(max_features=max_features)
    corpus.normalize()

def reduce(method, data, parameter):
    model.load_array(data)
    if method.lower() == "t-sne":
        model.t_sne(perplexity=parameter)
    elif method.lower() == "pca":
        model.reduce_dimensions_pca(components=parameter)
    else:
        print("Reduction method not recognized")

def cluster(method, clusters):
    if method.lower() == "kmeans":
        model.k_means(clusters=clusters)
    else:
        print("Clustering method not recognized")


def score(method):
    if method.lower() == "silhouette":
        model.record_silhouette_score()
    else:
        print("Scoring method not recognized")


def show():
    headlines = corpus.get_headlines()
    model.hover_plot(headlines)


corpus = CorpusCreator(directory="/Users/cameronlaedtke/PycharmProjects/MLPractice/Sentiment-Analysis/"
                            "Reddit-Sentiment-Analysis/data/")
model = WordsAnalysis(directory="/Users/cameronlaedtke/PycharmProjects/MLPractice/RedditNLP/data/")

load(file_name="the_donald/headlines.csv")
load(file_name="politics/headlines.csv")
load(file_name="The_Mueller/headlines.csv")

prep(max_features=2000)
reduce(method="pca", data=corpus, parameter=30)
reduce(method="t-sne", data=corpus, parameter=36)
cluster(method="kmeans", clusters=12)
score(method="silhouette")
show()

