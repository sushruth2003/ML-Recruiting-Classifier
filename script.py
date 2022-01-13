import pandas as pd
df = pd.read_csv('task.csv')
df = df[df['1 - no, 2 - maybe, 3 - yes, 4 - intern'].notna()]

# convert gpa column to floats


df['GPA'] = pd.to_numeric(df['GPA'],errors = 'coerce')

df2 = df.dropna(subset=['GPA'])
print((abs(len(df2)-len(df)))/len(df)*100) # percentage of missing values in dataset, 


from sklearn.impute import SimpleImputer
import numpy as np
imp = SimpleImputer(missing_values=np.nan, strategy = 'mean')
imp = imp.fit(df[['GPA']])
df['GPA'] = imp.transform(df[['GPA']]).ravel()
df2 = df.dropna(subset= ['GPA'])
print((abs(len(df2)-len(df)))/len(df)*100) # percentage of missing values in dataset, 
# Goal is to check that all nan values have been replaced by the mean value in the datset. 

df['hours']  = df['How much time can you commit per week?'] #making column easier to work with having such a wordy name is annoying
df = df.assign(hours = lambda x: x['hours'].str.extract('(\d+)')) # using regex to scrape out first number from the range/ single number each person gave. 
print(df['hours'])

# since dtype is still object, we should convert it to numerical data again for ease of use. 
df['hours'] = pd.to_numeric(df['hours'],errors = 'coerce')
df['hours']

imp = SimpleImputer(missing_values=np.nan, strategy = 'mean')
imp = imp.fit(df[['hours']])
df['hours'] = imp.transform(df[['hours']]).ravel()
df2 = df.dropna(subset= ['hours'])
print((abs(len(df2)-len(df)))/len(df)*100) # percentage of missing values in dataset, 
# Goal is to check that all nan values have been replaced by the mean value in the datset. 
df.drop(columns = ['How much time can you commit per week?']) # irrelevant now that we have the hours column
df['word count'] = (df['Why does this team interest you?'] + df['What value will you bring to Quant?'] + df['What do you hope to get out of Quant?']).apply(lambda x: len(str(x).split(" ")))
df['char count'] = (df['Why does this team interest you?'] + df['What value will you bring to Quant?'] + df['What do you hope to get out of Quant?']).str.len()
def avg_word(sentence):
  words = sentence.split()
  return (sum(len(word) for word in words)/len(words))
df['avg_word'] = (df['Why does this team interest you?'] + df['What value will you bring to Quant?'] + df['What do you hope to get out of Quant?']).apply(lambda x: avg_word(x))

from nltk.corpus import stopwords
stop = stopwords.words('english')
df['stopwords'] = (df['Why does this team interest you?'] + df['What value will you bring to Quant?'] + df['What do you hope to get out of Quant?']).apply(lambda x: len([x for x in x.split() if x in stop]))

def class_type(value):
    if value == 'Freshman':
        return 1
    if value == 'Sophomore':
        return 2
    if value == 'Junior':
        return 3
    if value == 'Senior':
        return 4
    return 0
df['class value'] = df['Year in School'].apply(lambda x: class_type(x))
df.head()

df['score'] = df['1 - no, 2 - maybe, 3 - yes, 4 - intern']

df = df.drop(['1 - no, 2 - maybe, 3 - yes, 4 - intern'], axis = 1)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def get_resume_score(text):
    cv = CountVectorizer(stop_words='english')
    count_matrix = cv.fit_transform(text)
    #Print the similarity scores
    #print("\nSimilarity Scores:")
     
    #get the match percentage
    matchPercentage = cosine_similarity(count_matrix)[0][1] * 100
    matchPercentage = round(matchPercentage, 2) # round to two decimal
    
    #print("Your resume matches about "+ str(matchPercentage)+ "% of the job description.")
    return matchPercentage

df['combined text answers'] = df['What do you hope to get out of Quant?'] + df['What value will you bring to Quant?']+ df['Why does this team interest you?']
swe_jd = "At Quant, we are seeking highly motivated students with a passion for code, speed, and an attention to coding best practices. As a member of the Software Development team, you will be utilizing the latest resources to help us design our in-house API and interface with exchanges world wide. You can expect to be challenged by balancing system designs with speed and reliability. We expect around at least 3-4 hours of work per week including team meetings. What to expect: Hone your skills in Software Development in an environment as tight and fast-paced as a real Quantitative Trading Fund Flexible projects that help build your skills and an environment that values individual contribution while maximizing for what you like to do Access to the cutting-edge of quant resources only available to members of Quant Learn about the intersection between research, trading infrastructure, and finance Requirements:  Knowledge of at least one programming language, preferably two or more. One must be from the following: C/C++, Java, Python, JavaScript/React, C#, Go, SQL and databases, … Think I missed something? No worries! Still send in your application and it will be reviewed on a case-by-case basis… we’re looking for people who are willing to learn above all else! Can devote at least 3-4 hours per week, with preference towards at least 4+ hours A passion for developing efficient, readable, and elegant code Experience with rapid development and using Git Willing to learn about new tech stacks and collaborate in team discussions Additional Preferences: CS 128, CS 225, CS 411, CS 357, CS 374, CS 233, and equivalents Our culture is focused on maximizing learning, and everything we do is towards this goal. If you think there’s a project we should be working on, please apply and help us grow our platform! We want to make sure the skills and projects you build will be an accelerating factor in your career. The platforms we build will be open-sourced and used within our RSO to help other teams develop the algorithms used in our fund. "
qr_jd = "The Quantitative Research division uses their understanding of finance, data science, statistical analysis, and mathematics to guide research on financial markets. The division develops hypotheses (algorithms) that are realized by the strategy implementation team. How to Join If you have experience with finance research, data science, or mathematics research, the Quantitative Research division may be right for you. Strong applicants will have experience in the following: Finance, economics, or econometric research Market microstructure, derivatives, or crypto assets Quantitative/statistical analysis using fundamental data, market data, or alternative data Data science"
si_jd = "The strategy implementation division applies data science techniques to quantitative finance as they draw correlations between different asset classes How to Join If you have experience in data engineering or data science, the strategy implementation division might be right for you. Strong applicants will have experience in the following: Data extraction, transformation, and load (ETL) processes Data acquisition through API access Python Machine learning techniques"
business_jd = "At Quant, we are seeking talented students to help spread our RSO’s brand and message. As a member of the business division you will help us build our relations with our corporate sponsors, planning and hosting events for students at UIUC, and running advertisement and recruiting campaigns to help others know more about what we do! There’s plenty of opportunity for you to take charge and lead your own campaigns, all with the support of a strong technical RSO to accomplish anything technical you might need. We expect around 3-4 hours of work per week including team meetings. What to expect: Network with corporate sponsors, recruiters, and engineers at top quant firms Lead your very own marketing campaign focused on a variety of topics from recruitment to events to Quad Day! Learn about the impact marketing roles have on Quant firms, while also delving into the world of finance through hosting talks with our corporate sponsors! Work with our technical teams to implement new ideas on website design, UI/UX design, and sponsor-only materia"
#clean_swe = clean_job_decsription(swe_jd)
##create_word_cloud(clean_swe)
#clean_qr = clean_job_decsription(qr_jd)
#create_word_cloud(clean_qr)
#clean_si = clean_job_decsription(si_jd)
##create_word_cloud(clean_si)
#clean_business = clean_job_decsription(business_jd)
#create_word_cloud(clean_business)

df['swe_score'] = 0
df['qr_score'] = 0
df['si_score'] = 0
df['business_score'] = 0
for index, row in df.iterrows():
    
    df.at[index, 'swe_score'] = get_resume_score([row['combined text answers'], swe_jd])
    df.at[index, 'qr_score'] = get_resume_score([row['combined text answers'], qr_jd])
    df.at[index, 'si_score'] = get_resume_score([row['combined text answers'], si_jd])
    df.at[index, 'business_score'] = get_resume_score([row['combined text answers'], business_jd])
#df['clean_qr_jd']= clean_job_decsription(qr_jd)
#df['clean_si_jd'] = clean_job_decsription(si_jd)
#df['clean_business_jd'] = clean_job_decsription(business_jd)

#df['swe_score'] = df.apply(lambda x: get_resume_score(df['combined text answers'], df['clean_swe_jd']))
# Train Test Split
from sklearn.model_selection import train_test_split
X = df[['GPA', 'word count', 'swe_score', 'qr_score', 'si_score', 'business_score']]
y = df['score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
model = SVC(kernel = "linear", C = 0.23, random_state = 102)
model.fit(X_train, y_train)
y_hat = model.predict(X) # so we can get predictions for all datapoints
acc = accuracy_score(y, y_hat)
print(acc)

prediction = pd.DataFrame(y_hat, columns = ['score']).to_csv('output.csv')
