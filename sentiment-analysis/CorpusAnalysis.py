import pickle
import numpy as np
import pandas as pd
import csv
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.text import Annotation
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabaz_score
import warnings
warnings.filterwarnings("ignore", message=": RuntimeWarning: numpy.dtype size changed")
from .CorpusCreation import *


class CorpusAnalyser:
    def __init__(self, directory):
        self.directory = directory

    def load_array_csv(self, array_file):
        file_path = self.directory + "BagOfWords/" + array_file
        fileObject = open(file_path, 'rb')
        self.X = pickle.load(fileObject)

    def load_data(self, array):
        self.X = array

    def load_headlines(self, file):
        file_path = self.directory + file
        self.raw_corpus = pd.read_csv(filepath_or_buffer=file_path, sep=",",
                                      names=["title", "author", "created_utc", "score",
                                             "link_flair_text", "domain", "self_text", "id"])
        self.headlines = self.raw_corpus.loc[:,"title"]
        print("Titles loaded")

    def headlines_to_file(self, to_file):
        to_file = self.directory + to_file
        self.headlines.to_csv(path_or_buf=to_file)

    def get_sample(self, size):
        self.X = self.X.sample(n=size)
        print("Reduced to sample of:"+size)

    def reduce_dimensions_pca(self, components):
        '''Dimensionality reduction using principle component analysis (PCA), produces lower dimensional array'''
        self.num_components = components
        if components <= 1:
            pca = PCA(n_components=components, svd_solver="full")
        else:
            pca = PCA(n_components=components)

        self.X = pca.fit_transform(self.X)
        print(self.X.shape)
        print('Dimensions reduced to: '+str(pca.n_components_))

    def k_means(self, clusters):
        '''K-means'''
        self.algorithm = "kmeans"
        kmeans = KMeans(n_clusters=clusters)

        self.X_dist_matrix = kmeans.fit_transform(self.X)
        self.labels = kmeans.labels_

        labels = ["X", "Y"]

        self.num_clusters = clusters
        self.df = pd.DataFrame(data=self.X, columns=labels)
        self.labels_df = pd.DataFrame(data=self.labels)
        self.df['labels'] = self.labels_df

        #self.X_labeled = np.append(self.X, self.labels, axis=1)
        print("Kmeans complete")

    def db_scan(self, eps, min_samples):
        '''DBSCAN clustering algorithm. Higher min_samples or lower eps
        indicate higher density necessary to form a cluster'''
        self.algorithm = "dbscan"
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(self.X)
        self.labels = db.labels_
        self.predicted_labels = db.fit_predict(self.X)
        n_clusters = len(set(self.labels))
        self.num_clusters = n_clusters
        n_core_samples = len(db.core_sample_indices_)
        components = db.components_
        n = 0
        # TODO: figure out how to make this label plot as grey
        for i in range(len(self.labels)):
            if(self.labels[i] == -1):
                n=n+1
                self.labels[i] = 1
                #print('-1 found')

        # self.df = pd.DataFrame(data=self.X, columns=labels)
        # self.labels_df = pd.DataFrame(data=self.labels)
        # self.df['labels'] = self.labels_df

        print('Number of noisy samples: '+str(n)+" of "+str(len(self.labels)))
        print('Clusters: ' + str(n_clusters))
        print('Number of core samples: '+str(n_core_samples))
        print("DBSCAN complete")

    def gaussian_mixture(self, n_components):
        '''Gaussian mixture clustering algorithm'''
        self.algorithm = "gaussian mixture"
        gm = GaussianMixture(n_components=n_components).fit(self.X)
        self.labels = gm.predict(self.X)
        unique, counts = np.unique(self.labels, return_counts=True)
        mydict = dict(zip(unique, counts))

        labels = ["X", "Y"]

        plt.bar(list(mydict.keys()), mydict.values(), color='g')
        plt.title("Gaussian Mixture")
        plt.ylabel("Number of skews")
        plt.xlabel("Cluster")
        #plt.show()

        print("bic: "+str(gm.bic(self.X)))
        #print("labels: "+str(self.labels))
        #print("Weights: "+str(gm.weights_))
        print("Score: "+str(gm.score(self.X)))
        print('Clusters: '+str(len(set(self.labels))))
        print('Converged: '+str(gm.converged_))
        print("Gaussian mixture complete")

    def bayesian_gaussian_mixture(self, n_components, weight_concentration_prior_type, weight_concentration_prior,
                                mean_precision_prior, n_init, max_iter, init_params):
        '''Bayesian Gaussian Mixture clustering algorithm. Low value for weight_concentration_prior will put more
        weight on a few components, high value will allow a larger number of components to be active in the mixture.'''
        bgm = BayesianGaussianMixture(n_components=n_components,
                                      weight_concentration_prior_type=weight_concentration_prior_type,
                                      weight_concentration_prior=weight_concentration_prior,
                                      mean_precision_prior=mean_precision_prior,
                                      n_init=n_init,
                                      max_iter=max_iter,
                                      init_params=init_params)
        bgm.fit(self.X)
        self.labels = bgm.predict(self.X)

        unique, counts = np.unique(self.labels, return_counts=True)
        mydict = dict(zip(unique, counts))
        print(mydict)

        plt.bar(list(mydict.keys()), mydict.values(), color = 'g')
        plt.ylabel("Number of skews")
        plt.xlabel("Cluster")
        plt.title(weight_concentration_prior_type)

        plt.gcf().text(0.05, 0.05, "Parameters initialized using: "+init_params)
        plt.gcf().text(0.05, 0.01, "Weight concentration prior: "+str(weight_concentration_prior))
        plt.gcf().text(0.7, 0.05, "Mean precision prior: "+str(mean_precision_prior))
        plt.gcf().text(0.7, 0.01, "Likelihood: "+str("%.2f"%bgm.lower_bound_))
        #plt.show()

        print("Weights: "+str(bgm.weights_))
        print("Converged: "+str(bgm.converged_))
        print("Number of iterations to reach convergence: "+str(bgm.n_iter_))
        print("Lower bound value on likelihood: "+str(bgm.lower_bound_))
        print("Bayesian Gaussian mixture complete")

    def t_sne(self, perplexity):
        self.algorithm = "t-sne"
        self.perplexity = perplexity
        '''TSNE clustering algorithm'''
        self.X = TSNE(n_components=2, perplexity=perplexity).fit_transform(self.X)

        print("TSNE complete")

    def get_cmap(self, n, name='hsv'):
        '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
        RGB color; the keyword argument name must be a standard mpl colormap name.'''
        return plt.cm.get_cmap(name, n)

    def plot(self):
        cmap = self.get_cmap(n=self.num_clusters)
        #cmap = self.get_cmap(n=10)


        plt.scatter(x=self.df.loc[:,"X"], y=self.df.loc[:, "Y"], s=0.5 ** 2, c=cmap(self.df.loc[:,"labels"]))
        # for index in enumerate(self.X_labeled):
        #     plt.scatter(x=index[1][0], y=index[1][1], s=0.5 ** 2, c=cmap(index[1][2]))
        #plt.scatter(x=self.X[:, 0], y=self.X[:, 1], s=0.5 ** 2, c=)
        plt.title("TSNE Embedding")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")

        plt.gcf().text(0.05, 0.03, "Perplexity: " + str(self.perplexity))
        plt.gcf().text(0.3, 0.03, "Clusters: " + str(self.num_clusters))
        plt.show()

    def hover_plot(self, titles):
        cmap = self.get_cmap(n=self.num_clusters)

        self.df["titles"] = titles

        fig, ax = plt.subplots()

        sc = plt.scatter(x=self.df.loc[:, "X"], y=self.df.loc[:, "Y"], s=0.5 ** 2, c=cmap(self.df.loc[:, "labels"]))

        annot = ax.annotate("", xy=(0, 0), xytext=(-200, 10), textcoords="offset points",
                            bbox=dict(boxstyle="round", fc="w"),
                            arrowprops=dict(arrowstyle="->"))

        annot.set_visible(False)

        def update_annot(ind):
            pos = sc.get_offsets()[ind["ind"][0]]
            annot.xy = pos
            text = "{}".format(" ".join([self.df.loc[n, "titles"] for n in ind["ind"]]))
            annot.set_text(text)
            annot.set_wrap(True)
            #annot.get_bbox_patch().set_facecolor(cmap(norm(c[ind["ind"][0]])))
            annot.get_bbox_patch().set_alpha(0.4)

        def hover(event):
            vis = annot.get_visible()
            if event.inaxes == ax:
                cont, ind = sc.contains(event)
                if cont:
                    update_annot(ind)
                    annot.set_visible(True)
                    fig.canvas.draw_idle()
                else:
                    if vis:
                        annot.set_visible(False)
                        fig.canvas.draw_idle()

        fig.canvas.mpl_connect("motion_notify_event", hover)

        plt.title("TSNE Embedding")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")

        plt.gcf().text(0.05, 0.03, "Perplexity: " + str(self.perplexity))
        plt.gcf().text(0.3, 0.03, "Clusters: " + str(self.num_clusters))

        plt.show()

    def record_silhouette_score(self):
        '''Testing with silhouette score. Cluster distance matrix based on centroid locations is needed.
         Doesn't work with DBSCAN, GaussianMixture'''
        self.silhouetteScore = silhouette_score(self.X_dist_matrix, self.labels)
        self.record_score(algorithm=self.algorithm, clusters=self.num_clusters, perplexity=self.perplexity,
                          score=self.silhouetteScore)

        print('Silhouette score: '+str(self.silhouetteScore))

    def get_calinski_harabaz_score(self):
        '''Defined as the ratio between the within-cluster dispersion and the between-cluster dispersion.'''
        self.CHScore = calinski_harabaz_score(self.X, self.labels)

        print('Calinski Harabas score: '+str(self.CHScore))

    def record_score(self, algorithm, clusters, perplexity, score):
        with open(self.directory+"BagOfWords/algorithm_scores.csv", "a", encoding="utf-8") as outfile_data:
            data = [algorithm, clusters, perplexity, score]
            writer = csv.writer(outfile_data, lineterminator='\n')
            writer.writerow(data)

    def clear_files(self, d_file):
        with open(self.directory + d_file, "w+") as file:
            file.close()