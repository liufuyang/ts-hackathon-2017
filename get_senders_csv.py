import boto3
import botocore
import csv

BUCKET_NAME = 'ts-dev-cs-training-data'

s3 = boto3.resource('s3').Bucket(BUCKET_NAME)

s3.download_file('hackathon_2017/senders/senders.csv', 'senders.csv')

with open('senders.csv', mode='r') as infile:
    reader = csv.reader(infile)
    doc_sender_dict = dict((rows[0],rows[1]) for rows in reader)
    print(len(doc_sender_dict))
    pickle.dump(doc_sender_dict, open('doc_sender_dict.pickle', 'wb'))