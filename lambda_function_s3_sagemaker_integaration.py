import json
import os
import pandas as pd
import boto3
import sys
import re
#import nltk
#nltk.download('punkt')

if sys.version_info[0] < 3:
    from StringIO import StringIO # Python 2.x
else:
    from io import StringIO # Python 3.x


aws_key = 'yourkey'
aws_secret = 'yourkeysec'
bucket_name = 'bucket-name'
object_key = 'folder1/filename.csv'

ENDPOINT_NAME = os.environ['ENDPOINT_NAME']
client_sm = boto3.client(service_name='sagemaker-runtime')

def load_data(aws_key=aws_key, aws_secret=aws_secret, bucket_name=bucket_name, object_key=object_key):
    client = boto3.client('s3', aws_access_key_id=aws_key,
        aws_secret_access_key=aws_secret)

    csv_obj = client.get_object(Bucket=bucket_name, Key=object_key)
    body = csv_obj['Body']
    csv_string = body.read().decode('utf-8')
    df = pd.read_csv(StringIO(csv_string), encoding='latin')
    df = df.dropna()
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)
    return df

def clean(text):
    return re.sub('[^a-zA-Z0-9]', ' ', text)

def pushToS3(df,name):
    csv_buffer = StringIO()
    df.to_csv(csv_buffer,index=False)
    s3 = boto3.client("s3",
                      aws_access_key_id=aws_key,
                      aws_secret_access_key=aws_secret)
    filename="folder1/outputfile/"+name+".csv"
    s3.put_object(Bucket=bucket_name, Body=csv_buffer.getvalue(), Key=filename)
    print("Done pushing to s3")

def lambda_handler(event, context):
    guidelines = load_data()
    print('Data loaded successfully')
    print(len(guidelines))
    guidelines['Guidelines'] = guidelines['Guidelines'].apply(lambda x: str(x))
    guidelines['Guidelines'] = guidelines['Guidelines'].apply(lambda x: x.lower())
    guide_lines = guidelines.iloc[:,2:3]
    guide_lines['Guidelines'] = guide_lines['Guidelines'].apply(lambda x:clean(x))
    print('cleaned guidelines')
    final_df = pd.DataFrame(columns=['__label__mask', '__label__social_distance', '__label__work_from_home',
       '__label__workforce_allowed_to_work', '__label__gloves_requirement',
       '__label__permission_reopen'])
    print('dummy df created')
    for i in range(len(guide_lines)):
        sentences = guide_lines.Guidelines[i]
#        tokenized_sentences = review_to_words(sentences)
#        payload = {"instances" : [tokenized_sentences], "configuration": {"k": 6}}
        tokenized_sentences = [sentences]
        payload = {"instances" : tokenized_sentences, "configuration": {"k": 6}}
        response = client_sm.invoke_endpoint(EndpointName=ENDPOINT_NAME,
                               Body=json.dumps(payload),
                               ContentType='application/json')
        #predictions = json.loads(response)
        predictions = json.loads(response['Body'].read().decode('utf-8'))
        temp_df = pd.DataFrame([predictions[0]['prob']], columns=predictions[0]['label'])
        #temp_df = pd.DataFrame([json.loads(response)[0]['prob']], columns=json.loads(response)[0]['label'])
        final_df = pd.concat([final_df, temp_df])
        final_df = final_df.reset_index(drop=True)
    print('preditions done for all data')
    final_result_df = pd.concat([guidelines, final_df], axis=1)
    file_name='guidelines_predicted_file'
    pushToS3(final_result_df,file_name)

    return {
        'statusCode': 200,
        'body': json.dumps('Successfully uploaded prediction results to S3')
        }
