'''
Script to scrape the top posts and comments data from a subreddit
'''
import time
import csv
import praw


class ScrapeReddit:

    def __init__(self, directory):
        self.directory = directory
        print('Authenticating...')
        self.reddit = praw.Reddit("CamsNLPBot", user_agent="web:cams-reddit-bot:v0.1 (by /u/cam_man_can)")
        print('Authenticated as /u/{}'.format(self.reddit.user.me()))

    def set_subreddit(self, subreddit):
        self.subreddit = subreddit
        self.headlines_file = self.directory + "/" + subreddit + "/headlines.csv"
        self.comments_file = self.directory + "/" + subreddit + "/comments.csv"

    def get_posts(self):
        '''Fetches the top 1000 posts from the past year from a specified subreddit.'''
        with open(self.headlines_file, "a",  encoding="utf-8") as outfile:
            for submission in self.reddit.subreddit(self.subreddit).top('year', limit=1000):
                print(submission.title)
                data = [
                    submission.title,
                    submission.author,
                    submission.created_utc,
                    submission.score,
                    submission.link_flair_text,
                    submission.domain,
                    "%r" % submission.selftext,
                    submission.id
                ]
                writer = csv.writer(outfile)
                writer.writerow(data)
                time.sleep(1)

    def get_comments(self):
        '''Gets the top 100 comments of each top 1000 post of subreddit'''
        print("Fetching comments")
        with open(self.headlines_file, "r", encoding="utf-8") as infile:
            reader = csv.reader(infile)
            resume_flag = 0
            row_counter = 0

            for row in reader:
                post_id = str(row[7])
                row_counter+=1

                if post_id == "6am00f":
                    resume_flag = 1
                # was originally if not resume_flag
                if resume_flag:
                    continue

                submission = self.reddit.submission(id=post_id)
                submission.comments.replace_more(limit=30, threshold=10)

                comment_count = 0

                for comment in submission.comments.list():
                    if comment_count >= 100:
                        break

                    if isinstance(comment, praw.models.MoreComments):
                        comment_count += 1
                        continue

                    comment_str = comment.body

                    comment_count += 1

                    if comment_str == "[deleted]" or comment_str == "[removed]":
                        continue

                    with open(self.comments_file, "a",  encoding="utf-8") as outfile:
                        writer = csv.writer(outfile)
                        writer.writerow(["%r" % comment_str, post_id])

            print("Finished post:", post_id, "Row count: ", row_counter)
            time.sleep(2)

    def clear_files(self):

        with open(self.headlines_file, "w+") as file:
            file.close()

        with open(self.comments_file, "w+") as file:
            file.close()






