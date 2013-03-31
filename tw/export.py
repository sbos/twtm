#exports data from mongodb to files

from sys import argv

if len(argv) < 2:
    print 'Usage: %s output_dir % (argv[0])'

from mongo import tweets
from os.path import join
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

lemmatizer = WordNetLemmatizer()
tokenizer = RegexpTokenizer('[^\w\']+', gaps=True)
en_stopwords = stopwords.words('english')

output_dir = argv[1]
users = tweets.distinct('user.screen_name')

word2num = {}
num2word = []

for user in users:
	user_feed = []
	for tweet in tweets.find({'user.screen_name': user}, fields={'text': 1}).sort( [('id', 1)] ):
		text = tweet['text']
		text = tokenizer.tokenize(text)
		words = [lemmatizer.lemmatize(w.lower().encode('utf8')) for w in text if not w in en_stopwords]

		num_repr = []
		for word in words:
			wid = word2num.get(word)
			if wid == None:
				wid = len(num2word) + 1
				word2num[word] = wid
				num2word.append(word)
			num_repr.append(wid)

		user_feed.append(num_repr)

	with open(join(output_dir, user), "w") as output:
		tweet_num = 1
		for num_repr in user_feed:
			for wid in num_repr:
				output.write('%d %d %d\n' % (tweet_num, wid, 1))
			tweet_num += 1
		print 'user %s processed, %d tweets' % (user, tweet_num)

with open(join(output_dir, "__num2word"), "w") as output:
	for w in xrange(len(num2word)):
		output.write('%s %s\n' % (w, num2word[w]))

	print 'num2word index written'