import pandas as pd
import re

def preprocess_pandas(data, columns):
    ''' <data> is a dataframe which contain  a <text> column  '''
    df_ = pd.DataFrame(columns=columns)
    df_ = data
    df_['comment_text'] = data['comment_text'].replace('[a-zA-Z0-9-_.]+@[a-zA-Z0-9-_.]+', '', regex=True)                      # remove emails
    df_['comment_text'] = data['comment_text'].replace('((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(\.|$)){4}', '', regex=True)    # remove IP address
    df_['comment_text'] = data['comment_text'].replace(r'http\S+', '', regex=True).replace(r'www\S+', '', regex=True)          # remove URLs
    df_['comment_text'] = data['comment_text'].str.replace('[#,@,&,<,>,\,/,-]','')                                             # remove special characters
    df_['comment_text'] = data['comment_text'].str.replace('[^\w\s#@/:%.,_-]', '', flags=re.UNICODE)                           # remove emojis+
    df_['comment_text'] = data['comment_text'].str.replace('[','')
    df_['comment_text'] = data['comment_text'].str.replace(']','')
    df_['comment_text'] = data['comment_text'].str.replace('\n', ' ')
    df_['comment_text'] = data['comment_text'].str.replace('\t', ' ')
    df_['comment_text'] = data['comment_text'].str.replace(' {2,}', ' ', regex=True)                                           # remove 2 or more spaces
    df_['comment_text'] = data['comment_text'].str.lower()
    df_['comment_text'] = data['comment_text'].str.strip()
    df_['comment_text'] = data['comment_text'].replace('\d', '', regex=True)                                                   # remove numbers
    df_.drop_duplicates(subset=['comment_text'], keep='first')
    df_ = df_.dropna()
    return df_

