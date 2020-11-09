# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 15:41:29 2020

@author: sukandulapati
"""

import json
import os
import pandas as pd
import boto3
import sys
import re
#import numpy as np
import psycopg2
import datetime as dt
#import numpy as np

if sys.version_info[0] < 3:
    from StringIO import StringIO # Python 2.x
else:
    from io import StringIO # Python 3.x

#parameters should be specified in the environment variables, lambda function 
ENDPOINT_NAME = os.environ['ENDPOINT_NAME']
table = os.environ['table']
bucket_name = os.environ['bucket_name']
object_key = os.environ['object_key']

#Recommended to control access by IAM roles, else you need following
aws_key = os.environ['aws_key']
aws_secret = os.environ['aws_secret']


client_sm = boto3.client(service_name='sagemaker-runtime')



def pushToS3(df,name):
    csv_buffer = StringIO()
    df.to_csv(csv_buffer,index=False)
    s3 = boto3.client("s3",
                      aws_access_key_id=aws_key,
                      aws_secret_access_key=aws_secret)
    filename="guidelines/outputfile/"+name+".csv"
    s3.put_object(Bucket=bucket_name, Body=csv_buffer.getvalue(), Key=filename)
    print("Done pushing to s3")



def load_data():
    con = psycopg2.connect(dbname='dev', host='redshift-cluster-safe-at-x.cqbfdwvyfklp.us-east-2.redshift.amazonaws.com', port='5439', user='awsadminuser', password='SafeXredshift123')
    cur = con.cursor()
    query = "SELECT * FROM " + table +";"
    #cur.execute("SELECT * FROM covid_data.huschblackwell_news;")
    cur.execute(query)
    dlist = list(cur.fetchall())
    df = pd.DataFrame(dlist)
    return df

def clean(text):
    text = re.sub('[^a-zA-Z0-9]', ' ', text)
    text = text.split()
    text = " ".join(text)
    return text

# def pushToRedshift(df1):
#     conn = psycopg2.connect(dbname='dev', host='redshift-cluster-safe-at-x.cqbfdwvyfklp.us-east-2.redshift.amazonaws.com', port='5439', user='awsadminuser', password='SafeXredshift123')
#     cur1 = conn.cursor()
#     for i in range(len(df1)):
#         cur1.execute("""insert into dw.ml_predicted_labels 
#                 (table_name,table_w_id,algorithm,label_type,label) 
#                 values(%s,%s,%s,%s,%s)""",(df1.table_name[i],int(df1.table_w_id[i]),df1.algorithm[i],df1.label_type[i],df1.label[i]))

#     cur1.close()
#     conn.commit()
#     conn.close()
#     print("Done pushing to Redshift")

def lambda_handler(event, context):
    guidelines = load_data()
    print('Data loaded successfully')
    guidelines.columns = ['w_id','location_w_id','datasource_id','source_day_w_id','g_value','Links','s_upsertTimestamp','w_upsertTimestamp']
    guidelines = guidelines[['w_id', 'g_value']]
    print(len(guidelines))
    guidelines['g_value'] = guidelines['g_value'].astype(str)
    guidelines['g_value'] = guidelines['g_value'].apply(lambda x: x.lower())
    guidelines['g_value'] = guidelines['g_value'].apply(lambda x: clean(x))
    guide_lines = guidelines[['w_id', 'g_value']]
    
    sentences = list(guide_lines.g_value)
    tokenized_sentences = sentences
    payload = {"instances" : tokenized_sentences}
    print('payload created')
    response = client_sm.invoke_endpoint(EndpointName=ENDPOINT_NAME,
                                Body=json.dumps(payload),
                                ContentType='application/json')
        
    print('response done')
    predictions = json.loads(response['Body'].read().decode('utf-8'))
    label_list = []
    for i in range(len(sentences)):
           label_list.append(predictions[i]['label'][0].split('__label__')[1])
    final_df = pd.DataFrame(label_list, columns=['label'])
    
    guide_lines['label'] = final_df['label']
    final_df = guide_lines[['w_id', 'label']]
    final_df.columns = ['table_w_id', 'label']
    
    final_df['table_name'] = table
    final_df['algorithm'] = 'new-labels-guidelines'
    final_df['label_type'] = 'categorization'
    final_df['w_upsert_timestamp'] = pd.Series([dt.datetime.now()] * len(final_df))
        
    final_df = final_df[['table_name', 'algorithm', 'table_w_id', 'label_type', 'label', 'w_upsert_timestamp']]
    
    print('prediction df is created')
    
    print('file is ready to push S3')
           
    #pushToRedshift(final_df)
    
    file_name='predicted_file_push_to_redshift_new_labels'
    pushToS3(final_df,file_name)

    return {
        'statusCode': 200,
        'body': json.dumps('Successfully uploaded prediction results to S3')
        }
