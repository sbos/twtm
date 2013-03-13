import pymongo

connection = pymongo.MongoClient()
db = connection.dataset
tweets = db['tweets']
