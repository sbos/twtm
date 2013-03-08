import urllib
import json
from mongo import tweets
from sys import maxint
from datetime import datetime, timedelta
from pytz import timezone

search_params = dict(q='#dreamindustries', rpp=100, result_type='recent',
    include_entities=True)

direction = "past"

while True:
    if direction == "past":
        last_tweet = tweets.find_one(fields={'id': 1},
            sort=[('id', 1)])

        if last_tweet == None or len(last_tweet) < 1:
            last_tweet = maxint
        else:
            last_tweet = last_tweet['id']
        print 'last loaded id: %d' % last_tweet

        search_params['max_id'] = last_tweet - 1
    else:
        raise "Oops, not implemented yet"

    stream = urllib.urlopen("http://search.twitter.com/search.json?"
        + urllib.urlencode(search_params))

    results = json.load(stream)['results']

    for tweet in results:
        for i in xrange(len(tweet['entities']['hashtags'])):
            tweet['entities']['hashtags'][i]['text'] = tweet['entities']['hashtags'][i]['text'].lower()

        tweet['created_at'] = datetime.strptime(tweet['created_at'], '%a, %d %b %Y %H:%M:%S +0000')
        tweet['created_at'] = timezone('Europe/Moscow').localize(tweet['created_at'])
        tweet['created_at'] = tweet['created_at'] - timedelta(hours=4)

        print tweet['created_at']

        tweets.insert(tweet)

    if len(results) < 1:
        break
