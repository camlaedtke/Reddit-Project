import matplotlib.pyplot as plt
from matplotlib import rc
import sys
import csv
import math
import numpy as np
import nltk as nltk
import nltk.sentiment.vader
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from fuzzywuzzy import fuzz

# TODO: Group comments and posts by user. Rank users by karma, polarity score. Most frequent users. Are they divisive?
# TODO: Things to try: put corpus into bag of words, try out different tools like n-grams, cosine similarity, t-SNE etc


class SentimentAnalyser:

    def __init__(self, directory, subreddit):
        data_fp = directory+subreddit
        self.directory = directory
        self.subreddit = subreddit
        self.posts_file = data_fp + "/headlines.csv"
        self.comments_file = data_fp + "/comments.csv"
        self.positive_file = data_fp + "/positive_list.txt"
        self.negative_file = data_fp + "/negative_list.txt"
        self.very_positive_file = data_fp + "/very_positive_list.txt"
        self.very_negative_file = data_fp + "/very_negative_list.txt"
        self.topics_positive_file = data_fp + "/topics/positive_list"
        self.topics_negative_file = data_fp + "/topics/negative_list"
        self.topics_very_positive_file = data_fp + "/topics/very_positive_list"
        self.topics_very_negative_file = data_fp + "/topics/very_negative_list"
        self.data_file = data_fp + "/topical_data.csv"
        self.topical_dir = data_fp + "/topics/"
        self.subreddits_file = directory + "/subreddit_data.csv"

    #def set_subreddit(self, subreddit):

    def run_sentiment_analysis(self):
        '''Reads the posts and comments and performs the analysis.'''
        sia = SIA()

        with open(self.posts_file, "r") as infile_posts:
            with open(self.comments_file, "r") as infile_comments:

                post_reader = csv.reader(infile_posts)
                comment_reader = csv.reader(infile_comments)

                row_count = 0
                positive_count = 0
                negative_count = 0
                very_positive_count = 0
                very_negative_count = 0
                total_count = 0

                # Looks at the post headline. Labels it positive or negative
                for row in post_reader:
                    results = sia.polarity_scores(row[0])
                    print("Analysing post rows: " + str(row_count + 1), end="\r")
                    row_count += 1
                    total_count += 1

                    if 0.6 > results['compound'] > 0.2:
                        with open(self.positive_file, "a", encoding="utf-8") as outfile_posts_0:
                            outfile_posts_0.write(row[0] + "\n")
                            positive_count += 1

                    elif results['compound'] > 0.6:
                        with open(self.very_positive_file, "a", encoding="utf-8") as outfile_posts_0:
                            outfile_posts_0.write(row[0] + "\n")
                            very_positive_count += 1

                    elif -0.6 < results['compound'] < -0.2:
                        with open(self.negative_file, "a", encoding="utf-8") as outfile_posts_1:
                            outfile_posts_1.write(row[0] + "\n")
                            negative_count += 1

                    elif results['compound'] < -0.6:
                        with open(self.very_negative_file, "a", encoding="utf-8") as outfile_posts_1:
                            outfile_posts_1.write(row[0] + "\n")
                            very_negative_count += 1
                print("\nDone.")

                row_count = 0

                # Looks at the comments. Labels them positive or negative
                for row in comment_reader:
                    results = sia.polarity_scores(row[0])
                    print("Analysing comment rows: " + str(row_count + 1), end="\r")
                    row_count += 1
                    total_count += 1

                    if 0.6 > results['compound'] > 0.2:
                        with open(self.positive_file, "a", encoding="utf-8") as outfile_comments_0:
                            outfile_comments_0.write(row[0] + "\n")
                            positive_count += 1

                    elif results['compound'] > 0.6:
                        with open(self.very_positive_file, "a", encoding="utf-8") as outfile_comments_0:
                            outfile_comments_0.write(row[0] + "\n")
                            very_positive_count += 1

                    elif -0.6 < results['compound'] < -0.2:
                        with open(self.negative_file, "a", encoding="utf-8") as outfile_comments_1:
                            outfile_comments_1.write(row[0] + "\n")
                            negative_count += 1

                    elif results['compound'] < -0.6:
                        with open(self.very_negative_file, "a", encoding="utf-8") as outfile_comments_1:
                            outfile_comments_1.write(row[0] + "\n")
                            very_negative_count += 1

                neutral_count = total_count - positive_count - negative_count - very_positive_count - very_negative_count

                neutral = neutral_count / total_count * 100
                neutral = round(neutral, 2)
                positive = positive_count / total_count * 100
                positive = round(positive, 2)
                negative = negative_count / total_count * 100
                negative = round(negative, 2)
                very_positive = very_positive_count / total_count * 100
                very_positive = round(very_positive, 2)
                very_negative = very_negative_count / total_count * 100
                very_negative = round(very_negative, 2)
                # Add data to a file so that we don't have to repeat everytime
                with open(self.data_file, "a", encoding="utf-8") as outfile_data:
                    data = [self.subreddit, very_negative, negative, neutral, positive, very_positive]
                    writer = csv.writer(outfile_data, lineterminator='\n')
                    writer.writerow(data)

                with open(self.subreddits_file, "a", encoding="utf-8") as outfile_data:
                    data = [self.subreddit, very_negative, negative, neutral, positive, very_positive]
                    writer = csv.writer(outfile_data, lineterminator='\n')
                    writer.writerow(data)

                print("\nDone.")

    def word_distribution(self, filename):
        '''Plots words distributions for a given file.'''

        file_path = self.directory + filename

        tokenizer = RegexpTokenizer(r'\w+')
        stop_words = set(stopwords.words('english'))
        all_words = []

        with open(file_path, "r", encoding='utf-8') as f_pos:
            for line in f_pos.readlines():
                words = tokenizer.tokenize(line)
                for w in words:
                    if w.lower() not in stop_words and w.lower() != "n":
                        all_words.append(w.lower())

        result = nltk.FreqDist(all_words)
        print(result.most_common(8))

        plt.style.use('ggplot')

        y_val = [x[1] for x in result.most_common(len(all_words))]
        y_final = []
        for i, k, z, t in zip(y_val[0::4], y_val[1::4], y_val[2::4], y_val[3::4]):
            y_final.append(math.log(i + k + z + t))
        x_val = [math.log(i + 1) for i in range(len(y_final))]

        plt.xlabel("Words (Log)")
        plt.ylabel("Frequency (Log)")
        plt.title("Word Frequency Distribution")
        plt.plot(x_val, y_final)
        plt.show()

    def run_topical_analysis(self, string):
        '''Searches for a user given string and performs sentiment analysis for it.'''
        sia = SIA()

        print("Running topical analysis for " + string + "...")
        with open(self.posts_file, "r") as infile_posts:
            with open(self.comments_file, "r") as infile_comments:
                post_reader = csv.reader(infile_posts)
                comment_reader = csv.reader(infile_comments)

                include_list = []
                positive_count = 0
                very_positive_count = 0
                negative_count = 0
                very_negative_count = 0
                total_count = 0

                test_post_counter = 0
                test_comment_counter = 0

                for row in post_reader:
                    print("Analyzing posts and comment rows: " + str(total_count + 1), end="\r")

                    test_post_counter += 1

                    match_post_title = fuzz.partial_ratio(string, row[0])
                    match_post_flair = fuzz.partial_ratio(string, row[4])
                    if row[6] == "''":
                        match_post_selftext = 0
                        if match_post_flair >= 85 or match_post_title >= 85:
                            result_0 = sia.polarity_scores(row[0])
                            result_1 = None
                            include_list.append(row[7])
                        else:
                            continue
                    else:
                        match_post_selftext = fuzz.partial_ratio(string, row[6])
                        if match_post_flair >= 85 or match_post_title >= 85 or match_post_selftext >= 85:
                            result_0 = sia.polarity_scores(row[0])
                            result_1 = sia.polarity_scores(row[4])
                            include_list.append(row[7])
                        else:
                            continue

                    if 0.6 > result_0['compound'] > 0.2:
                        with open(self.topics_positive_file + "_" + "%r" % string + r".txt", "a",
                                  encoding="utf-8") as outfile_posts:
                            outfile_posts.write(row[0] + "\n")
                            positive_count += 1
                            total_count += 1

                    elif result_0['compound'] > 0.6:
                        with open(self.topics_very_positive_file + "_" + "%r" % string + r".txt", "a",
                                  encoding="utf-8") as outfile_posts:
                            outfile_posts.write(row[0] + "\n")
                            very_positive_count += 1
                            total_count += 1

                    elif -0.6 < result_0['compound'] < -0.2:
                        with open(self.topics_negative_file + "_" + "%r" % string + r".txt", "a",
                                  encoding="utf-8") as outfile_posts:
                            outfile_posts.write(row[0] + "\n")
                            negative_count += 1
                            total_count += 1

                    elif result_0['compound'] < -0.6:
                        with open(self.topics_very_negative_file + "_" + "%r" % string + r".txt", "a",
                                  encoding="utf-8") as outfile_posts:
                            outfile_posts.write(row[0] + "\n")
                            very_negative_count += 1
                            total_count += 1

                    if result_1 is not None:
                        if 0.6 > result_1['compound'] > 0.2:
                            with open(self.topics_positive_file + "_" + "%r" % string + r".txt", "a",
                                      encoding="utf-8") as outfile_posts:
                                outfile_posts.write(row[0] + "\n")
                                positive_count += 1

                        elif result_1['compound'] > 0.6:
                            with open(self.topics_very_positive_file + "_" + "%r" % string + r".txt", "a",
                                      encoding="utf-8") as outfile_posts:
                                outfile_posts.write(row[0] + "\n")
                                very_positive_count += 1

                        elif -0.6 < result_1['compound'] < -0.2:
                            with open(self.topics_negative_file + "_" + "%r" % string + r".txt", "a",
                                      encoding="utf-8") as outfile_posts:
                                outfile_posts.write(row[0] + "\n")
                                negative_count += 1

                        elif result_1['compound'] < -0.6:
                            with open(self.topics_very_negative_file + "_" + "%r" % string + r".txt", "a",
                                      encoding="utf-8") as outfile_posts:
                                outfile_posts.write(row[0] + "\n")
                                very_negative_count += 1

                for row in comment_reader:
                    print("Analyzing posts and comment rows: " + str(total_count + 1), end="\r")

                    test_comment_counter += 1

                    if row[1] in include_list:
                        total_count += 1

                        result = sia.polarity_scores(row[0])

                        if 0.6 > result['compound'] > 0.2:
                            with open(self.topics_positive_file + "_" + "%r" % string + r".txt", "a",
                                      encoding="utf-8") as outfile_comments:
                                outfile_comments.write(row[0] + "\n")
                                positive_count += 1

                        elif result['compound'] > 0.6:
                            with open(self.topics_very_positive_file + "_" + "%r" % string + r".txt",
                                      "a", encoding="utf-8") as outfile_comments:
                                outfile_comments.write(row[0] + "\n")
                                very_positive_count += 1

                        elif -0.6 < result['compound'] < -0.2:
                            with open(self.topics_negative_file + "_" + "%r" % string + r".txt", "a",
                                      encoding="utf-8") as outfile_comments:
                                outfile_comments.write(row[0] + "\n")
                                negative_count += 1

                        elif result['compound'] < -0.6:
                            with open(self.topics_very_negative_file + "_" + "%r" % string + r".txt",
                                      "a", encoding="utf-8") as outfile_comments:
                                outfile_comments.write(row[0] + "\n")
                                very_negative_count += 1

                neutral_count = total_count - positive_count - negative_count - very_positive_count - very_negative_count

                neutral = neutral_count / total_count * 100
                neutral = round(neutral, 2)
                positive = positive_count / total_count * 100
                positive = round(positive, 2)
                negative = negative_count / total_count * 100
                negative = round(negative, 2)
                very_positive = very_positive_count / total_count * 100
                very_positive = round(very_positive, 2)
                very_negative = very_negative_count / total_count * 100
                very_negative = round(very_negative, 2)
                # Add data to a file so that we don't have to repeat everytime
                with open(self.data_file, "a", encoding="utf-8") as outfile_data:
                    data = [string, very_negative, negative, neutral, positive, very_positive]
                    writer = csv.writer(outfile_data, lineterminator='\n')
                    writer.writerow(data)

                print(neutral_count)
                print(positive_count)
                print(negative_count)
                print(very_positive_count)
                print(very_negative_count)
                print()
                print(test_post_counter)
                print(test_comment_counter)
                #plot_word_types(total_count, negative_count, positive_count, string)
                #stacked_word_types(total_count, negative_count, positive_count, string)

    def stacked_bar(self, group):
        if group == "topics":
            data_file = self.data_file
        elif group == "subreddits":
            data_file = self.subreddits_file
        else:
            print("not recognized")

        data = []
        with open(data_file, "r") as infile_data:
            data_reader = csv.reader(infile_data)
            for row in data_reader:
                data.append(row)


        plt.figure(figsize= (10, 6))
        series_labels = ['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive']
        colors = ['#960000', '#f94f4f', '#efefef', '#76fc76', '#007000']
        category_labels = []
        X = []

        for row in data:
            category_labels.append(row[0])
            X.append(row[1:])

        print(X)
        X = list(map(list, zip(*X)))


        for i in range(len(X)):
            for j in range(len(X[i])):
                X[i][j] = float(X[i][j])

        y_label = "Percent"

        ny = len(X[0])
        ind = list(range(ny))

        axes = []
        cum_size = np.zeros(ny)

        X = np.array(X)
        bar_width = 0.3

        for i, row_data in enumerate(X):
            axes.append(plt.bar(ind, row_data, bottom=cum_size, color=colors[i], width=bar_width, label=series_labels[i]))
            cum_size = np.add(cum_size, row_data, out=cum_size, casting='unsafe')

        plt.xticks(ind, category_labels)
        plt.ylabel(y_label)
        plt.legend()
        plt.show()

    def clear_files(self):

        with open(self.positive_file, "w+") as file:
            file.close()

        with open(self.negative_file, "w+") as file:
            file.close()

        with open(self.very_positive_file, "w+") as file:
            file.close()

        with open(self.very_negative_file, "w+") as file:
            file.close()





