import logging
from TwitterFactCheck import TwitterFactCheck
import pandas as pd
import logging
from datetime import *
from multiprocessing.dummy import Process
from collections import Counter

logging.basicConfig(level=logging.INFO)


class SusUserSearcher(object):
    def __init__(
            self, api: TwitterFactCheck, df_tweets: pd.DataFrame,
            df_users: pd.DataFrame):
        """A class to get information about suspicious users

        Args:
            api (TwitterFactCheck): The API instance to run query
            df_tweets (pd.DataFrame): A Pandas DataFrame containing all fetched suspicious user's tweets
            df_users (pd.DataFrame): A Pandas DataFrame containing all fetched suspicious user's information
        """
        self.api: TwitterFactCheck = api
        self.df_tweets: pd.DataFrame = df_tweets
        self.df_users: pd.DataFrame = df_users

    def query_tweets(
        user1: pd.Series, user2: pd.Series, start_time: str, end_time: str
    ) -> pd.DataFrame:
        """Get all interactive tweets between two users

        Args:
            user1 (pd.Series): The information about first user
            user1 (pd.Series): The information about second user
            end_time (str): The end time for update
            method (str): The search method for the original users

        Returns:
            (pd.DataFrame): All tweets between user1 and user2
        """
        data = list()
        if user1.name > user2.name:  # compare based on user id
            user1, user2 = user2, user1
        query = " OR ".join(
            [
                "(@{user1} @{user2})",
                "(from:{user1} @{user2})",
                "(from:{user2} @{user1})",
                "(from:{user2} to:{user1})",
                "(from:{user1} to:{user2})",
            ]
        )
        query = query.format(user1=user1.username, user2=user2.username)
        tweets = api.search_tweets(
            query, start_time=start_time, end_time=end_time)
        tweets["user1"], tweets["user2"] = user1.name, user2.name
        tweets["author1"], tweets["author2"] = user1.username, user2.username
        data.append(tweets)
        return pd.concat(data) if data else pd.DataFrame()

    def get_updates(self, status: dict, start_time: str,
                    end_time: str, method: str):
        """Get the updates of suspicious users and their tweets based on current data

        Args:
            status (dict): A dictionary to store the data we've fetched
            start_time (str): The start time for update
            end_time (str): The end time for update
            method (str): The search method for the original users
        """

        df_method_tweets = self.df_tweets[self.df_tweets["method"] == method]
        df_method_users = self.df_users[self.df_users["method"] == method]
        iteration = status["iteration"] = df_method_users["iteration"].max(
        ) + 1
        status["df_method_users"], status["df_method_tweets"] = (
            df_method_users,
            df_method_tweets,
        )
        status["status"] = "prepare"
        df_sus_users = pd.Series(Counter(df_method_tweets["author_id"]))
        df_sus_users = df_sus_users[df_sus_users > 5]
        df_sus_users = df_sus_users[df_sus_users > df_sus_users.quantile(0.8)]
        status["df_new_users"] = df_sus_users
        df_new_users = dict()
        status["status"] = "fetching {} new users".format(len(df_sus_users))
        for author_id in df_sus_users.index.astype(int):
            if author_id not in df_method_users.index:
                info = api.search_user(author_id)
                if info:
                    df_new_users[author_id] = info
        status["df_new_users"] = df_new_users = (
            pd.DataFrame(df_new_users).T.dropna(how="all").sort_index()
        )
        status["df_new_tweets"] = df_new_tweets = list()
        status["status"] = "fetching tweets"
        for i in range(len(df_new_users)):
            status["i"] = i
            for t in range(0, len(df_method_users)):
                status["t"] = t
                df_new_tweets.append(
                    self.query_tweets(
                        df_new_users.iloc[i],
                        df_method_users.iloc[t],
                        start_time,
                        end_time,
                    )
                )

            for j in range(i + 1, len(df_new_users)):
                df_new_tweets.append(
                    self.query_tweets(
                        df_new_users.iloc[i],
                        df_new_users.iloc[j],
                        start_time, end_time))
                status["j"] = j

        df_new_tweets = pd.concat(df_new_tweets).set_index(["id"])
        df_new_tweets.text = df_new_tweets.text.replace("\\s+", " ")
        df_new_tweets["content"] = df_new_tweets.text.str.replace(
            "(@[\\w|\\d]+|\\#[\\w|\\d]+|https\\S+)", " "
        )
        for s in ["\\s{2,}", "RT:\\s?", "^\\s+\\$"]:
            df_new_tweets["content"] = df_new_tweets["content"].str.replace(
                s, "")
        status["status"] = "fetching complete"
        for col in ["author_id"]:
            df_new_tweets[col] = df_new_tweets[col].astype(int)
        status["df_method_tweets"] = df_new_tweets
        df_new_tweets["method"], df_new_users["method"] = method, method
        df_new_tweets["iteration"] = df_new_users["iteration"] = iteration


if __name__ == "__main__":
    api = TwitterFactCheck.TwitterFactCheck("Backbackup.json")
    df_tweets = pd.read_csv(
        "Data/Network/NetworkTweets.csv",
        sep="\t",
        index_col="id")
    df_users = pd.read_csv(
        "Data/Network/NetworkUsers.csv", index_col="user_id", sep="\t"
    )
    start_time, end_time = "20220101", "20220401"

    searcher = SusUserSearcher(api, df_tweets, df_users)

    status = dict()
    method = "Headline"
    searcher.get_updates(status, start_time, end_time, method)

    df_new_users = status["df_new_users"]
    df_new_tweets = status["df_new_tweets"]
    i, j, t = status["i"], status["j"], status["t"]
    df_new_users.to_csv(
        "Data/tmp/{}_{}_{}_NetworkUsers.csv".format(i, t, j),
        sep="\t",
        index_label="user_id",
    )
    df_new_tweets.to_csv(
        "Data/tmp/{}_{}_{}_NetworkTweets.csv".format(i, t, j),
        sep="\t",
        index_label="id",
    )
