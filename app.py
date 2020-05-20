
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import nltk
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
import string
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

# Add title
st.title( 'Impeachment Speech Analysis' )

# Load list of house of representatives members on 3/4/2020
reps = pd.read_excel( 'House-of-Representatives-2020-03-04.xlsx', sheet_name=0 )
reps['Last Name'] = reps['Member'].apply( lambda x: x.split( ' ' )[-1:][0].upper() )
gap = reps.iloc[0,0][7]
reps['State'] = reps['District'].apply( lambda x: x.split( gap )[0].upper() )
# st.write( reps.head() )

# Load computer speech transcripts extracted previously from gov sharing site
dates = [ '2019-11-13', '2019-11-14' ]
speech = pd.DataFrame()
for date in dates:
    tmp = pd.read_excel( date+'.xlsx', sheet_name=0 )
    tmp['Date'] = pd.to_datetime( date )
    speech = speech.append( tmp, ignore_index=True )
speech = speech[['Date','Time','Speaker','Text']]
# speech.info()

# Compute last name of each speaker
speech['Last Name'] = speech['Speaker'].apply( lambda x: x.split( ' ' )[-1:][0].upper() )
speech.loc[speech['Last Name'].isin(['TEMPORE','CLERK',']','CHAIR','SECRETARY']),'Last Name'] = np.nan
# st.write( speech.head() )

# Create function for guessing who's who.
@st.cache
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
# st.write( speech['Guesses'].value_counts() )

# Merge based on those guesses
merged = pd.merge( speech, reps, left_on='Guesses', right_index=True, how='inner' )

# Filter based on user selection
st.sidebar.markdown( '## Data filter' )
ordered_reps = merged.groupby( 'Member' ).agg( 'count' ) \
    .sort_values( 'Date', ascending=False )[['Date']]
ordered_reps.columns = [ 'num_speeches' ]
special_options = [
    'All Representatives',
    'All Democrats',
    'All Republicans'
]
options = special_options + list( ordered_reps.index )
def name_to_index ( name ):
    return reps['Member'].tolist().index( name )
def show_rep_text ( text ):
    if text == 'All Representatives':
        return f"{text} - {len( merged )} speeches"
    if text == 'All Democrats':
        return f"{text} - {sum( merged.Party == 'Democratic' )} speeches"
    if text == 'All Republicans':
        return f"{text} - {sum( merged.Party == 'Republican' )} speeches"
    index = name_to_index( text )
    row = reps.iloc[index,:]
    party = row.Party if type( row.Party ) == str else '?'
    num = ordered_reps.loc[text].num_speeches
    return f"{row.Member} ({party[:3]}, {row.State}) " \
           f"- {num} speech{'' if num == 1 else 'es'}"
filter_text = st.sidebar.selectbox(
    'Analyze the speech(es) of which representative(s)?',
    options, index=0, format_func=show_rep_text
)
if filter_text == 'All Representatives':
    filtered = merged
elif filter_text == 'All Democrats':
    filtered = merged[merged.Party == 'Democratic']
elif filter_text == 'All Republicans':
    filtered = merged[merged.Party == 'Republican']
else:
    filtered = merged[merged.Member == filter_text]

# Word filter
include_adjectives_only = \
    st.sidebar.checkbox( 'Include high-impact adjectives only', True )

# Lemmatize and tokenize speeches to keep only important words
stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()
@st.cache
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
def simplify ( text ):
    return ' '.join( tokenize( text ) ) if type( text ) == str else np.nan
filtered['Simple Text'] = filtered['Text'].apply( simplify )

# Display filtered results if the user asked us to
st.sidebar.markdown( '## Outputs' )
if st.sidebar.checkbox( 'Show sample of selected speeches', True ):
    st.write( f"## Sample of {show_rep_text( filter_text )}" )
    if len( filtered ) > 0:
        st.write( 'Hover mouse over the text of a speech for details.' )
        for_display = filtered.head( 10 ).reset_index()
        if filter_text == 'All Representatives':
            columns = ['Date','Time','Member','Party','State','Text']
        elif filter_text in [ 'All Republicans', 'All Democrats' ]:
            columns = ['Date','Time','Member','State','Text']
        else:
            columns = ['Date','Time','Text']
        for_display = for_display[columns]
        st.write( for_display )
    else:
        st.write( "(No speeches)" )

# How Republican is a word?
republican_rows = merged.Party == 'Republican'
total_num_words = len( ' '.join( list( merged.Text ) ).split( ' ' ) )
rep_num_words = len( ' '.join( list( merged[republican_rows].Text ) ).split( ' ' ) )
midway_point = 1 - rep_num_words / total_num_words
def num_occurrences ( word, text ):
    return len( [ x for x in text.upper().split( ' ' ) if x == word.upper() ] )
@st.cache
def word_partisanship ( word ):
    count = lambda speech : num_occurrences( word, speech )
    occurrences = sum( merged.Text.apply( count ) )
    if occurrences == 0:
        return 0.5
    rep_occ = sum( merged[republican_rows].Text.apply( count ) )
    ratio = rep_occ / occurrences
    if ratio > midway_point:
        return ( ( ratio - midway_point ) / ( 1 - midway_point ) + 1 ) / 2
    else:
        return ratio / ( 2 * midway_point )
def word_color ( word, *args, **kwargs ):
    how_republican = word_partisanship( word )
    as_byte = int( ( how_republican * 2 - 1 ) * 255 )
    return (as_byte,0,255-as_byte)

# Keep only the high-impact words?
all_words = ' '.join( [ t for t in list( filtered['Simple Text'] )
    if type( t ) == str ] ).upper().split( ' ' )
adjectives_only = ''
if include_adjectives_only:
    important_words = [ 'american', 'united', 'national', 'bipartisan',
        'senate', 'unanimous', 'federal', 'social', 'bible', 'military',
        'strong', 'political', 'rural', 'financial', 'chinese', 'foreign',
        'partisan', 'freedom', 'mexico', 'global', 'black', 'impeachment',
        'daca', 'commitment', 'true', 'supreme', 'international', 'white',
        'difficult', 'fair', 'ukraine', 'serious', 'extraneous', 'necessary',
        'wrong', 'personal', 'honorable' ]
    all_words = [ w for w in all_words if w.lower() in important_words ]
    adjectives_only = '(Showing high-impact adjectives only.)'

# Add a word cloud
if st.sidebar.checkbox( 'Show wordcloud', True ):
    st.write( f"## Word cloud for {show_rep_text( filter_text )}" )
    st.write( 'Words colored blue are more spoken by Democrats; '
              'words colored more red are more spoken by Republicans.' )
    st.write( adjectives_only )
    if len( filtered ) > 0:
        wordcloud = WordCloud(
            width=600, height=400,
            background_color='white', #colormap='Dark2',
            color_func=word_color,
            max_words=50, collocations=False
        ).generate( ' '.join( all_words ) )
        plt.imshow( wordcloud, interpolation='bilinear' )
        plt.axis( 'off' )
        plt.show()
        st.pyplot()
    else:
        st.write( "(No speeches)" )

# Which words are said the most?
word_counts = pd.Series( all_words ).value_counts()
def nice_count_table ( word_list ):
    as_series = pd.Series( word_list )
    count_series = as_series.value_counts()
    return pd.DataFrame( {
        'Word' : count_series.index,
        'Count' : count_series.values
    } )
if st.sidebar.checkbox( 'Show top 25 words used', False ):
    st.write( f"## Top 25 words by {show_rep_text( filter_text )}" )
    st.write( adjectives_only )
    if len( filtered ) > 0:
        for_display = nice_count_table( all_words ).head( 25 )
        st.write( for_display )
    else:
        st.write( "(No speeches)" )

st.sidebar.markdown( '## Credits' )
st.sidebar.markdown( '''
Speech data comes from computer-generated transcripts of speeches on the floor
of the United States House of Representatives on November 13 and 14, 2019.
The data is provided [freely online](https://live.house.gov/?date=2019-11-13)
by the U.S. government and has not been checked for transcription accuracy.
''' )
st.sidebar.markdown(
    'Code by [Nathan Carter](http://nathancarter.github.io)' )
