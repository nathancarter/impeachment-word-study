#!/usr/bin/env python
# coding: utf-8

# # Mining the impeachment transcripts

# In[1]:


import pandas as pd
import numpy as np


# ## Load data
# 
# The files are named by date, `2019-11-13.xlsx` and upwards, as shown in the list below.
# 
# We load them and combine them into a single DataFrame.

# In[2]:


dates = [ '2019-11-13', '2019-11-14', '2019-11-15', '2019-11-18', '2019-11-19', '2019-11-20', '2019-11-21' ]
speech = pd.DataFrame()
for date in dates:
    tmp = pd.read_excel( date+'.xlsx', sheet_name=0 )
    tmp['Date'] = pd.to_datetime( date )
    speech = speech.append( tmp, ignore_index=True )
speech = speech[['Date','Time','Speaker','Text']]
speech.info()


# (Due to a current limitation in DeepNote, we must wrap `speech.head()` in a `print` statement.)

# In[3]:


speech.head()


# ## Last Names
# 
# We compute last names by extracting them from the speaker's name, and discarding words known to not be a last name.
# 
# TO DO: There are definitely some other cleanup steps that ought to be done, as you can see further below, but this is a good start.

# In[4]:


speech['Last Name'] = speech['Speaker'].apply( lambda x: x.split( ' ' )[-1:][0].upper() )
speech.loc[speech['Last Name'].isin(['TEMPORE','CLERK',']','CHAIR','SECRETARY']),'Last Name'] = np.nan
speech.head()


# In[5]:


speech['Last Name'].unique()


# ## Get representative names
# 
# The representatives are stored in a table extracted from Wikipedia.  Let's load that now.

# In[6]:


reps = pd.read_excel( 'House-of-Representatives-2020-03-04.xlsx', sheet_name=0 )
reps['Last Name'] = reps['Member'].apply( lambda x: x.split( ' ' )[-1:][0].upper() )
gap = reps.iloc[0,0][7]
reps['State'] = reps['District'].apply( lambda x: x.split( gap )[0].upper() )
reps.head()


# ## Joining the tables
# 
# To join these tables, we need to match up representative names in the one with the other.  Unfortunately, they are not always unique.
# 
# So we try to match up representative names and double-check them against the state mentioned in the introductory comments before the speaker begins speaking.

# In[7]:


def guess_rep_index ( i ):
    last_name = speech.loc[i,'Last Name']
    intro = speech.loc[i-1,'Text'][-100:] if i > 0 and type( speech.loc[i-1,'Text'] ) == str else ''
    matches = [ j for j in list( reps.index ) if reps.loc[j,'Last Name'] == last_name ]
    if len( matches ) == 1:
        return matches[0]
    for m in matches:
        if reps.loc[m,'State'] in intro:
            return m
    return np.nan
speech['Guesses'] = [ guess_rep_index( i ) for i in speech.index ]
speech['Guesses'].value_counts()


# This seems to successfully classify over $\frac13$ of the speakers:

# In[8]:


speech['Guesses'].isnull().value_counts()


# ## Process speech content
# 
# Load the tools needed for natural language processing.

# In[9]:


import nltk
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')


# Lemmatize and tokenize speeches to get the most important words only, and in canonical forms.

# In[10]:


from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
import string
from nltk.corpus import stopwords

stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

def tokenize ( text ):
    for char in string.punctuation:
        text = text.replace( char, ' ' )
    result = [ ]
    words = [ w for w in text.split( ' ' ) if w != '' ]
    for token, tag in pos_tag(words):
        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'
        token = lemmatizer.lemmatize(token, pos)
        if len(token) > 0 and token.lower() not in stop_words:
            result.append(token.lower())
    return result

# tokenize( 'You are one of the best kids I\'ve ever met, you know that?' )
### ==> ['one', 'best', 'kid', 'ever', 'meet', 'know']

def simplify ( text ):
    return ' '.join( tokenize( text ) ) if type( text ) == str else np.nan


# In[11]:


speech['Simple Text'] = speech['Text'].apply( simplify )
speech.head()


# In[12]:


all_words = ' '.join( [ t for t in list( speech['Simple Text'] ) if type( t ) == str ] ).split( ' ' )
word_counts = pd.Series( all_words ).value_counts()
word_counts


# In[13]:


tagged_words = pos_tag( [ w for w in word_counts.index if len( w ) > 0 ] )
tagged_adjs = [ pair[0] for pair in tagged_words if pair[1][:2] == 'JJ' ]
', '.join(list(word_counts[tagged_adjs].index[:200]))


# ## Merge tables

# In[14]:


merged = pd.merge( speech, reps, left_on='Guesses', right_index=True, how='inner' )
merged.head()


# In[15]:


len(merged)


# ## Add features

# In[16]:


merged['Text Length'] = merged['Text'].apply( lambda x: len(x) if type(x)==str else 0 )
merged.head()


# ## Try random things now

# In[17]:


dem_words = [ ]
rep_words = [ ]
def check_word_party ( word ):
    global dem_words, rep_words
    where_word = merged['Simple Text'].apply( lambda text: type(text) == str and word in text )
    results = merged[where_word]['Party'].value_counts()
    dem = results['Democratic'] if 'Democratic' in results.index else 0
    rep = results['Republican'] if 'Republican' in results.index else 0
    if dem > 1.5*rep:
        dem_words.append( word )
    if rep > 1.5*dem:
        rep_words.append( word )
# important_words = [ 'american', 'united', 'national', 'bipartisan', 'senate', 'unanimous', 'federal', 'social',
#     'bible', 'military', 'strong', 'political', 'rural', 'financial', 'chinese', 'foreign', 'partisan', 'freedom',
#     'mexico', 'global', 'black', 'impeachment', 'daca', 'commitment', 'true', 'supreme', 'international', 'white',
#     'difficult', 'fair', 'ukraine', 'serious', 'extraneous', 'necessary', 'wrong', 'personal', 'honorable' ]
common_adjs = word_counts[tagged_adjs].index[word_counts[tagged_adjs] > 50]
for word in common_adjs:
    check_word_party( word )
print( '%d Democratic words:' % len(dem_words) )
print( dem_words )
print( '%d Republican words:' % len(rep_words) )
print( rep_words )


# In[18]:


import matplotlib.pyplot as plt


# In[19]:


plt.hist( [ merged[merged['Party']=='Democratic']['Text Length'],
            merged[merged['Party']=='Republican']['Text Length'] ], label=['Dem','Rep'] )
plt.legend()
plt.yscale( 'log' )
plt.show()


# In[20]:


sum( merged[merged['Party']=='Democratic']['Text Length'] ), sum( merged[merged['Party']=='Republican']['Text Length'] )


# In[21]:


len( merged[merged['Party']=='Democratic']['Text Length'] ), len( merged[merged['Party']=='Republican']['Text Length'] )


# In[22]:


merged[merged['Party']=='Democratic']['Text Length'].mean(), merged[merged['Party']=='Republican']['Text Length'].mean()


# In[23]:


merged.info()


# In[24]:


merged['Simple assumed'] = merged['Assumed office'].apply( lambda x: str(x)[:4] ).astype( int )
data = merged.groupby( 'Member' )[['Simple assumed','Text Length']].agg({'Simple assumed':'first','Text Length':'sum'})
plt.scatter( x=data['Simple assumed'], y=data['Text Length'], alpha=0.3 )
plt.yscale( 'log' )
plt.show()


# In[25]:


data = merged.groupby( 'Member' )[['Simple assumed','Text Length']].agg({'Simple assumed':'first','Text Length':'mean'})
plt.scatter( x=data['Simple assumed'], y=data['Text Length'], alpha=0.3 )
plt.yscale( 'log' )
plt.show()


# ## Get time from speech?
# 
# Turns out you can't seem to do this.  I tried here but found only 36 of the thousands of texts had any detectable times.

# In[26]:


def get_time_in_seconds ( important_words ):
    if type( important_words ) != str:
        return None
    important_words = important_words.split( ' ' )
    if 'minute' in important_words:
        minute_index = len( important_words ) - 1 - important_words[::-1].index( 'minute' )
    else:
        minute_index = None
    if 'second' in important_words:
        second_index = len( important_words ) - 1 - important_words[::-1].index( 'second' )
    else:
        second_index = None
    if minute_index is None and second_index is None:
        return None
    if minute_index is None:
        index = second_index
    elif second_index is None:
        index = minute_index
    else:
        index = max( minute_index, second_index )
    if index == 0:
        return None
    unit = important_words[index]
    seconds = 60 if unit == 'minute' else 1
    quantity = important_words[index-1]
    conversion = { 'one' : 1, 'five' : 5, 'two' : 2, 'three' : 3, '30' : 30, 'four' : 4, '20' : 20,
                   '50' : 50, '90' : 90, '15' : 15, '45' : 45, 'a' : 1 }
    min_reasonable = 15  # nobody got <30sec to talk ("one second" == figure of speech)
    max_reasonable = 300 # nobody got >5min to talk ("30 minutes" == time of a recess)
    if quantity in conversion:
        result = conversion[quantity] * seconds
        if min_reasonable <= result <= max_reasonable:
            return result
    return None
merged['Simple Text'].apply( get_time_in_seconds ).value_counts()


# What if we try with the original text?  Seems better...you get 106 now.  Still only 10% of speeches.

# In[27]:


def get_time_in_seconds2 ( all_words ):
    if type( all_words ) != str:
        return None
    for char in string.punctuation:
        all_words = all_words.replace( char, ' ' )
    all_words = all_words.replace( 'MINUTES', 'MINUTE' )
    all_words = all_words.replace( 'SECONDS', 'SECOND' )
    all_words = all_words.lower().split( ' ' )
    if 'minute' in all_words:
        minute_index = len( all_words ) - 1 - all_words[::-1].index( 'minute' )
    else:
        minute_index = None
    if 'second' in all_words:
        second_index = len( all_words ) - 1 - all_words[::-1].index( 'second' )
    else:
        second_index = None
    if minute_index is None and second_index is None:
        return None
    if minute_index is None:
        index = second_index
    elif second_index is None:
        index = minute_index
    else:
        index = max( minute_index, second_index )
    if index == 0:
        return None
    unit = all_words[index]
    seconds = 60 if unit == 'minute' else 1
    quantity = all_words[index-1]
    conversion = { 'one' : 1, 'five' : 5, 'two' : 2, 'three' : 3, '30' : 30, 'four' : 4, '20' : 20,
                   '50' : 50, '90' : 90, '15' : 15, '45' : 45, 'a' : 1 }
    min_reasonable = 15  # nobody got <30sec to talk ("one second" == figure of speech)
    max_reasonable = 300 # nobody got >5min to talk ("30 minutes" == time of a recess)
    if quantity in conversion:
        result = conversion[quantity] * seconds
        if min_reasonable <= result <= max_reasonable:
            return result
    return None
merged['Text'].apply( get_time_in_seconds2 ).value_counts()


# In[ ]:




