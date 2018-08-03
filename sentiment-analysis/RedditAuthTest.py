import praw
from praw.models import MoreComments


reddit = praw.Reddit("CamsNLPBot", user_agent="web:cams-reddit-bot:v0.1 (by /u/cam_man_can)")
print('Authenticated as /u/{}'.format(reddit.user.me()))

#subreddit = reddit.subreddit("politics")
#print(subreddit.title)

submission = reddit.submission(id='3g1jfi')

#for top_level_comment in submission.comments:
 #   print(top_level_comment.body)


for top_level_comment in submission.comments:
    if isinstance(top_level_comment, MoreComments):
        continue
    print(top_level_comment.body)