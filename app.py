
# Import data science fundamentals
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import dashboard tools
import streamlit as st

# Import text processing tools
import nltk
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
import string
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

# Add title
st.title( 'Impeachment Speech Analysis' )

# Load list of House of Representatives members on 3/4/2020
reps = pd.read_excel( 'House-of-Representatives-2020-03-04.xlsx', sheet_name=0 )
# Split out last name and state into new columns
reps['Last Name'] = reps['Member'].apply( lambda x: x.split( ' ' )[-1:][0].upper() )
gap = reps.iloc[0,0][7]
reps['State'] = reps['District'].apply( lambda x: x.split( gap )[0].upper() )

# Load computer speech transcripts extracted previously
# from https://live.house.gov/?date=2019-11-13
#  and https://live.house.gov/?date=2019-11-14
dates = [ '2019-11-13', '2019-11-14' ]
speech = pd.DataFrame()
for date in dates:
    tmp = pd.read_excel( date+'.xlsx', sheet_name=0 )
    tmp['Date'] = pd.to_datetime( date )
    speech = speech.append( tmp, ignore_index=True )
speech = speech[['Date','Time','Speaker','Text']]

# Compute last name of each speaker in transcript dataset
speech['Last Name'] = speech['Speaker'].apply( lambda x: x.split( ' ' )[-1:][0].upper() )
# Remove words that are clearly not last names
speech.loc[speech['Last Name'].isin(['TEMPORE','CLERK',']','CHAIR','SECRETARY']),'Last Name'] = np.nan

# Create function for guessing which representative gave each speech
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
# Apply that function to the entire speeches dataset.
speech['Guesses'] = [ guess_rep_index( i ) for i in speech.index ]

# Merge the two datasets based on those guesses.
merged = pd.merge( speech, reps, left_on='Guesses', right_index=True, how='inner' )

# Filter the merged data based on the user's inputs
st.sidebar.markdown( '## Data filter' )
# Construct list of options in descending order of size
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
# Create a function that renders names nicely for showing in the list
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
# Show the drop-down list to the user
filter_text = st.sidebar.selectbox(
    'Analyze the speech(es) of which representative(s)?',
    options, index=0, format_func=show_rep_text
)
# Filter the merged dataset based on their choice
if filter_text == 'All Representatives':
    filtered = merged
elif filter_text == 'All Democrats':
    filtered = merged[merged.Party == 'Democratic']
elif filter_text == 'All Republicans':
    filtered = merged[merged.Party == 'Republican']
else:
    filtered = merged[merged.Member == filter_text]

# Ask the user whether we should pay attention only to major words
include_adjectives_only = \
    st.sidebar.checkbox( 'Include high-impact adjectives only', True )

# Use NLTK to extract just the interesting words and to stem them
stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()
# This function turns words into canonical forms
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
# This function runs the previous one across a big string of text
def simplify ( text ):
    return ' '.join( tokenize( text ) ) if type( text ) == str else np.nan
# Run that on the whole set of speeches
filtered['Simple Text'] = filtered['Text'].apply( simplify )

# Display filtered speech results if the user asked us to
st.sidebar.markdown( '## Outputs' )
if st.sidebar.checkbox( 'Show sample of selected speeches', True ):
    st.write( f"## Sample of {show_rep_text( filter_text )}" )
    if len( filtered ) > 0:
        # This prints a table of speeches, leaving out columns if they would
        # contain only one value, based on the user's chosen filter.
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
        # This shouldn't happen, but just in case, it's here:
        st.write( "(No speeches)" )

# How Republican is a word?
republican_rows = merged.Party == 'Republican'
total_num_words = len( ' '.join( list( merged.Text ) ).split( ' ' ) )
rep_num_words = len( ' '.join( list( merged[republican_rows].Text ) ).split( ' ' ) )
midway_point = 1 - rep_num_words / total_num_words
def num_occurrences ( word, text ):
    return len( [ x for x in text.upper().split( ' ' ) if x == word.upper() ] )
# The following function returns 0 for words spoken only by Democrats and 1 for
# words spoken only by Republicans, and varying levels in between for others.
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
# This function uses the previous to color Democratic words blue and Republican
# words red.  The return values are (R,G,B) color triples.
def word_color ( word, *args, **kwargs ):
    how_republican = word_partisanship( word )
    as_byte = int( ( how_republican * 2 - 1 ) * 255 )
    return (as_byte,0,255-as_byte)

# Choose which words we will study; first get all words in all speeches:
all_words = ' '.join( [ t for t in list( filtered['Simple Text'] )
    if type( t ) == str ] ).upper().split( ' ' )
# If the user asked for only high impact adjectives (which we list below, from
# having inspected the data in another analysis) then filter for just those.
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

# Add a word cloud if the user asked us to
if st.sidebar.checkbox( 'Show wordcloud', True ):
    st.write( f"## Word cloud for {show_rep_text( filter_text )}" )
    st.write( 'Words colored blue are more spoken by Democrats; '
              'words colored more red are more spoken by Republicans.' )
    st.write( adjectives_only )
    if len( filtered ) > 0:
        # This uses the Python wordcloud package
        wordcloud = WordCloud(
            width=600, height=400,
            background_color='white', #colormap='Dark2',
            color_func=word_color,
            max_words=50, collocations=False
        ).generate( ' '.join( all_words ) )
        # It creates images, which we show with Matplotlib
        plt.imshow( wordcloud, interpolation='bilinear' )
        plt.axis( 'off' )
        plt.show()
        st.pyplot()
    else:
        # This shouldn't happen, but just in case:
        st.write( "(No speeches)" )

# If the user asked for a table of the most-used words, produce it now:
word_counts = pd.Series( all_words ).value_counts()
# This function just takes a value_counts() and makes it look nice.
def nice_count_table ( word_list ):
    as_series = pd.Series( word_list )
    count_series = as_series.value_counts()
    return pd.DataFrame( {
        'Word' : count_series.index,
        'Count' : count_series.values
    } )
# If the user checked the checkbox...
if st.sidebar.checkbox( 'Show top 25 words used', False ):
    st.write( f"## Top 25 words by {show_rep_text( filter_text )}" )
    st.write( adjectives_only )
    if len( filtered ) > 0:
        # ...then show the table
        for_display = nice_count_table( all_words ).head( 25 )
        st.write( for_display )
    else:
        # This shouldn't happen, but just in case:
        st.write( "(No speeches)" )

# Add credits information to the sidebar, with links to sources.
st.sidebar.markdown( '## Credits' )
st.sidebar.markdown( '''
Speech data comes from computer-generated transcripts of speeches on the floor
of the United States House of Representatives on November 13 and 14, 2019.
The data is provided [freely online](https://live.house.gov/?date=2019-11-13)
by the U.S. government and has not been checked for transcription accuracy.
''' )
st.sidebar.markdown( 'App by [Nathan Carter](http://nathancarter.github.io)' )
st.sidebar.markdown( '[Code on GitHub](http://github.com/nathancarter/)' )
