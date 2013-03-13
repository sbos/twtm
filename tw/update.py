#downloads target user's tweets

import urllib
import json
from mongo import tweets
from sys import maxint, argv
from datetime import datetime, timedelta
from pytz import timezone
from twitter import *
from twconf import auth

if len(argv) < 2:
    print 'Usage: %s target_user [max_count=3200]' % (argv[0])

target_user = argv[1]

search_params = dict(count=200, result_type='recent',
    include_entities=True, include_rts=False, exclude_replies=True)
search_params['screen_name'] = target_user
max_count = 3200
if len(argv) > 2:
    max_count = float(argv[2])

direction = "past"

twitter = Twitter(auth=auth)

while True:
    tweet_count = tweets.find({'user.screen_name': target_user}).count()
    print 'tweets in database for user %s: %d' % (target_user, tweet_count)
    if tweet_count > max_count:
        print 'max_count %d reached. stopping' % (max_count)
        break

    if direction == "past":
        last_tweet = tweets.find_one({'user.screen_name': target_user}, 
            fields={'id': 1}, sort=[('id', 1)])

        if last_tweet == None or len(last_tweet) < 1:
            last_tweet = maxint
        else:
            last_tweet = last_tweet['id']
        print 'last loaded id: %d' % last_tweet

        search_params['max_id'] = last_tweet - 1
    else:
        raise "Oops, not implemented yet"

    results = twitter.statuses.user_timeline(**search_params)

    for tweet in results:
        for i in xrange(len(tweet['entities']['hashtags'])):
            tweet['entities']['hashtags'][i]['text'] = tweet['entities']['hashtags'][i]['text'].lower()

        # tweet['created_at'] = datetime.strptime(tweet['created_at'], '%a, %d %b %Y %H:%M:%S +0000')
        # tweet['created_at'] = timezone('Europe/Moscow').localize(tweet['created_at'])
        # tweet['created_at'] = tweet['created_at'] - timedelta(hours=4)

        print tweet['created_at']

        tweets.insert(tweet)

    if len(results) < 1:
        break
