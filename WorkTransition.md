# Panda Work Transition

### 1. Scrap the misinformation check cases

##### Scrap PolitiFact.com

1. Use the script and python class in <u>PolitiFact.py</u> to scrap the latest 20 pages of misinformation check from PolitiFact.com
2. The scrapped data will be saved to <u>Data/FactChecks/PolitiFact.csv</u> with corresponding meta information for the fact check like date, checkurl, statement, rate, etc.

##### Scrap Univision.com

1. Save the html of Univision  misinformation webpage in <u>Websites/Unvision.html</u>
2. Use python class defined in <u>Univison.py</u> to parse the webpage and save the fact check in <u>Data/FactChecks/Univision.csv</u> with similar format as Data/FactChecks/PolitiFact.csv

### 2. Use Twitter API to find tweets related to the fact checks

##### TwitterAPI.py

1. register an academic API account from twitter API, the keys are stored in TwitterAPI.json
2. Use class defined in <u>TwitterAPI.py</u> and <u>TwitterFactCheck.py</u> to run queries based on Twitter API. <u>TwitterFactCheck.py</u> also provides more helper method to get relevant information.

##### TwitterSearch.ipynb

1. Read the misinformation check cases from <u>Data/FactChecks/PolitiFact.csv</u> and <u>Data/FactChecks/Univision.csv</u> and normalize these test cases into a new dataframe.
2. Each row of the dataframe is a misinformation case, where we can extract three different kind of keyword for query
   - Headline of the check case, such as statement
   - Related URL to the check case, such as original webpage
   - All noun entities in the headline. The entities are extracted through POS tagging, such as Michigan/president, etc. 

3. For each of kind of query, I put them into the Twitter API and get all relevant tweets.
4. Store all the relevant tweets into <u>Data/FactChecks/FactTweets.csv</u> with relevant query keyword and fact check headline.

### 3. Use network methods and tweets related to fact checks to find out suspicious users on Twitter.

##### UserSearch.py

1. We define suspicious users as twitters users who has a published a lot of tweets or who has a lot of interaction with other suspicious users. The interaction can be reply, mentioning, etc.
2. The suspicious tweets are all relevant tweets related to fact check cases stored in <u>Data/FactChecks/FactTweets.csv</u> or tweets from other suspicious users.
3. We start with the tweets from the fact check tweets to find the suspicious users who are the authors of these tweets. Then we get all tweets related to the suspicious users, and again fine new authors. We can set a threshold to define who are the suspicious users (e.g. authored more than 10 suspicious tweets).
4. After many iterations, we stored all the collected tweets and users in <u>Data/Network/NetworkTweets.csv</u> and <u>Data/Network/NetworkUsers.csv</u>

##### GeoCommunity.ipynb

1. Read the user information from <u>Data/Network/NetworkUsers.csv</u> and draw the geo distribution 
2. Construct a graph where each vertex is a user and each edge weight is the number of tweets
3. Use the greedy  modularity to find the community within the users, and find the most popular user

### 4. Get the candidates tweets

##### CandCollector.py

1. Scrap ballotpedia.org and collect the information of all candidates who participated in the US election.
2. The candidates' data are exported to <u>Data/Candidates/Candidates.csv</u> with their running status, position, state, twitter account, etc.

##### CandTweets.ipynb

1. Read the candidates information from <u>Data/Candidates/Candidates.csv</u> and call Twitter API to fetch all tweets coming from or mentioning candidates' twitter account
2. Classify the tweets by their source, e.g. whether it's from our collected suspicious users, or from suspicious domain defined in the <u>Data/WebsiteCredbility.csv</u>
3. We export the count of these tweets into <u>Data/Candidates/CandTweetsCount.csv</u>, and export some sample tweets to  <u>Data/Candidates/SusDomainTweets.csv</u> and <u>Data/Candidates/SusUserTweets.csv</u>
4. Set a larger time range for tweets search so that we can get as much tweets as possible.

##### PeakDetect.py

1. For each candidates, we construct the daily count of fetched relevant tweets and turn it into time series data.
2. Use iqr method defined in PeakDetect.py to detect the peak of the time series. If the count is larger than 1.5 / 3 / 4 times of iqr, we will report the peak and have manual check on what happen on these days.
3. Run the same peak detection method on tweets only from suspicious user and suspicious domain.
4. Check the performance of peak detection by compare the days of high iqr and days when we have a misinformation check case.
5. We can also manually sample some tweets and look at what they're talking about. 

### 5. Machine learning tweet check

##### NLP.py

1. Read tweets from previous sampled tweets from <u>Data/Candidates/</u>. We can label the tweets by whether they're from suspicious domain or users.
2. Use different models defined in PredModel to predict whether a given tweet is from suspicious domain or users.
   - Logistic Regression: We use bag of words to construct the bag of words vector, and use logistic regression to do classification
   - LSTM / RNN: We use word2vec to turn each tweets into a large distributional representation, and feed the vector to the time series neuro network for training.
   - Emotion detection: We apply a emotional detection model from huggingface.com to get the emotion vector of each tweet, which can be anger / fear / neutral / etc. Then we pass the emotion vector into a linear / logistic regression / neuro network to see its performance.
3. The model's performance satisfying right now, because there are too much noise in the tweets, but we're looking into other models and solution as well, e.g. using the interaction count (like count, etc) as feature for detection.

##### PU/PU.ipynb

1. This is a project from a [paper](https://arxiv.org/pdf/2003.02736.pdf) working on detecting if a given text is worth for checking. We can get pertained PU model using run.sh after cloning the [repo](https://github.com/copenlu/check-worthiness-pu-learning).
2. Run.sh will called the python scripted by the paper's author to train a dozens of models able to determine if a tweet is worth for checking for a specific topic. The dataset for training and testing is [PHEME]([PHEME dataset for Rumour Detection and Veracity Classification (figshare.com)](https://figshare.com/articles/dataset/PHEME_dataset_for_Rumour_Detection_and_Veracity_Classification/6392078))
3. Then we try to apply the models on our tweets and manually look into whether it gives us a satisfying result. Unfortunately, for now these models are only able to detect tweets related to the topics.