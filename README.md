# Reddit Opinion Mining, Classification, and Sentiment Analysis

A project written in R and Python to mine a Reddit corpus and analyse Reddit sentiment.
Some code is taken from https://github.com/pranau97/reddit-opinion-mining and 
https://github.com/minimaxir/reddit-graph. 

## Requirements

### Python and its dependencies

1. Python 3
2. PRAW
3. requests
4. bs4
5. numpy
6. fuzzywuzzy
7. nltk
8. matplotlib

**virtual environment recommended**

### R and its dependencies

1. R
2. sna
3. ggnetwork
4. svglite
5. igraph
6. intergraph
7. rsvg
8. ggplot2

## Ongoing Projects

### Network Graph Visualization

Using the code from the network-graph directory, I would like to make a full network graph visualization of reddit 
subreddits. If I can find a way to mine post and/or comment data from hundreds of subreddits, I could map sentiment 
towards a given topic onto the network graph. It would be a really cool way to gauge the overall Reddit sentiment 
towards a controversial topic, and see how sentiment changes within Reddit sub-communities.

### Analyse Political Opinions 

Mine a large corpus of posts and comments from politically charged subreddits like r/the_donald, r/the_mueller,
r/democrats, and r/republican. Once I've mined the relevant data I could look graph the sentiment within each 
subreddit towards various controversial political figures and topics.

### Clustering of Reddit Users

Associate each Reddit user with it's own corpus and then apply various clustering algorithms to see if there 
are any intrinsic clusters of Reddit users that can be looked at through sentiment analysis.




