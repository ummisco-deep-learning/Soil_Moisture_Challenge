import os
import numpy as np
import pandas as pd

root = os.path.dirname(os.path.abspath(__file__))
sample_df = pd.read_csv(os.path.join(root,'data','SampleSubmission.csv'))
new_df = sample_df.copy()

def generate_new_submission(liste,name='newSubmission'):
    new_df['Values']=liste
    new_df.to_csv(os.path.join(root,'data','{}.csv'.format(name)))


def generate_random_submission():
    random_liste = np.random.random(5782)
    random_liste = [int(i*100) for i in random_liste]
    generate_new_submission(random_liste,'randomSubmission')

# generate_random_submission()
