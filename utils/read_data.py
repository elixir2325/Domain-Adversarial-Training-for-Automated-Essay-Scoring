import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def scale_down_score_df(df):

    def scale_down_score(df_group):
        score_range = {
            1:(2,12),
            2:(1,6),
            3:(0,3),
            4:(0,3),
            5:(0,4),
            6:(0,4),
            7:(0,30),
            8:(0,60),
        }
        prompt_id = df_group.name
        min_score = score_range[prompt_id][0]
        max_score = score_range[prompt_id][1]
        df_group['score_scaled'] = (df_group['score'] - min_score) / (max_score - min_score)
        return df_group

    df = df.groupby('essay_set').apply(lambda x: scale_down_score(x))

    return df

def read_data(source_prompt_id, target_prompt_id, config):

    df = pd.read_excel(config.data_path)
    df = df[['essay_id', 'essay_set', 'essay', 'domain1_score']]
    # rename columns
    df.columns = ['essay_id', 'essay_set', 'essay', 'score']

    df.dropna(subset=['score'], axis=0, inplace=True)
    df = scale_down_score_df(df) # add a column "score_scaled" which normalize the score to [0-1] scale based on the score range of each prompt
 
    df_source = df[df['essay_set']==source_prompt_id].copy(deep=True)
    df_target = df[df['essay_set']==target_prompt_id].copy(deep=True)

    df_train = df_source.sample(frac=0.9, random_state=42)
    df_val = df_source.drop(df_train.index)
    
    return df_train, df_val, df_target





