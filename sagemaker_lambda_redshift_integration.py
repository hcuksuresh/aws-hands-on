import json
import os
import pandas as pd
import boto3
import re
import psycopg2


ENDPOINT_NAME = os.environ['ENDPOINT_NAME']
client_sm = boto3.client(service_name='sagemaker-runtime')

def load_data():
    con = psycopg2.connect(dbname='dbname', host='hostname.redshift.amazonaws.com', port='portnumber', user='username', password='password')
    cur = con.cursor()
    cur.execute("SELECT * FROM dw.fact_littler_county_reopening_orders;")
    dlist = list(cur.fetchall())
    df = pd.DataFrame(dlist)
    return df

def clean(text):
    return re.sub('[^a-zA-Z0-9]', ' ', text)

def pushToRedshift(df1):
    conn = psycopg2.connect(dbname='dbname', host='hostname.redshift.amazonaws.com', port='portnumber', user='username', password='password')
    cur1 = conn.cursor()
    for i in range(len(df1)):
        cur1.execute("""insert into db.tablename 
                (column1,column2,column3,column4,column5,label) 
                values(%s,%s,%s,%s,%s,%s)""",(df1.column1[i],df1.column2[i],df1.column3[i],df1.column4[i],df1.column5[i],df1.label[i]))
    cur1.close()
    conn.commit()
    conn.close()
    print("Done pushing to Redshift")

def lambda_handler(event, context):
    guidelines = load_data()
    guidelines.iloc[:,2] = guidelines.iloc[:,2].apply(lambda x: str(x))
    guidelines.iloc[:,2] = guidelines.iloc[:,2].apply(lambda x: x.lower())
    guide_lines = guidelines.iloc[:,2:3]
    guide_lines.columns = ['Guidelines']
    guide_lines['Guidelines'] = guide_lines['Guidelines'].apply(lambda x:clean(x))
    final_df = pd.DataFrame(columns=['label'])
    
    for i in range(len(guide_lines)):
        sentences = guide_lines.Guidelines[i]
        tokenized_sentences = [sentences]
        payload = {"instances" : tokenized_sentences}
        response = client_sm.invoke_endpoint(EndpointName=ENDPOINT_NAME,
                               Body=json.dumps(payload),
                               ContentType='application/json')
        predictions = json.loads(response['Body'].read().decode('utf-8'))
        #prob1 = str(predictions[0]['prob'][0])
        label1 = str(predictions[0]['label'][0])
        label1 = label1.split('__label__')[1]
        temp_df = pd.DataFrame([label1], columns=['label'])
        final_df = pd.concat([final_df, temp_df])
        final_df = final_df.reset_index(drop=True)
      
    print('preditions done for all data')
    guidelines.columns = ['column1', 'column2', 'column3', 'column4', 'column5']
    
    cln = list(guidelines.columns) + list(final_df.columns)
    final_result_df = pd.concat([guidelines, final_df], axis=1)
    final_result_df.columns = cln
    pushToRedshift(final_result_df)

    return {
        'statusCode': 200,
        'body': json.dumps('Successfully uploaded prediction results to Redshift')
        }
