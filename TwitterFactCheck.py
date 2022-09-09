from datetime import datetime, timedelta
import json
import logging
import re
from typing import Any, Dict, Union, Set
import traceback

import altair as alt
import pandas as pd

from TwitterAPI import TwitterAPI


class DomainChecker(object):
    def __init__(
            self, df_domain: Union[str, pd.DataFrame],
            true_domain: Union[str, Set]):
        '''Initialize the domain checker

        Args:
            df_domain: The dataframe contains the domain information
            true_doamin: A set contains the true domain
        '''
        if isinstance(df_domain, str):
            df_domain = pd.read_csv(df_domain, sep="\t")

        self.fake_domain = set(
            df_domain[(df_domain["fake"] == 1) | (df_domain["lowcred"] == 1)]
            ["domain"])
        if isinstance(true_domain, str):
            with open(true_domain, "r") as f:
                true_domain = set(json.loads(f.read()))
                true_domain = true_domain | set(
                    df_domain[df_domain["traditional"] == 1]["domain"])
        self.true_domain = true_domain

    def check(self, text: str) -> int:
        '''Check the credibility of domain in a given text

        Args:
            text: The given text

        Returns:
            0 to 2, higher value means higher credibility
        '''
        for domain in self.true_domain:
            if domain.lower() in text:
                return 2

        for domain in self.fake_domain:
            if domain.lower() in text:
                return 0

        return 1


class TwitterFactCheck(TwitterAPI):
    def __init__(self, api_tokens: Union[Dict[str, str], str], ):
        super().__init__(api_tokens)

    @staticmethod
    def parse_fact(fact: pd.Series):
        def parse_time(text: str):
            month, day, year = None, None, None
            for t in (
                re.findall(
                    r"[A-Z]\w+ \d+, \d+",
                    text)[0].replace(
                    ",",
                    "").split(" ")
            ):
                if len(t) == 4 and t.isdigit():
                    year = int(t)
                elif t.isdigit():
                    day = int(t)
                else:
                    month = datetime.strptime(t, "%B").month
            return datetime(year, month, day)

        fact = dict(fact)
        fact["FactDate"] = parse_time(fact["Setting"])
        fact["CheckDate"] = parse_time(fact["CheckTime"])
        fact["Tags"] = fact["Tags"].split(";")
        return fact

    def analyze_fact(self, fact: pd.Series, period: int = 7,
                     search: bool = False):
        fact = self.parse_fact(fact)
        start_time = fact["FactDate"] - timedelta(period)
        end_time = min(fact["CheckDate"] + timedelta(period), datetime.now())
        query = (
            " OR ".join(
                state[1: -1]
                for state in re.findall('[“"].+[”"]', fact["Statement"])) +
            " lang:en")
        logging.info(
            "Count tweets for query {} from {} to {}".format(
                query, start_time, end_time
            )
        )
        counts = self.count_tweets(
            query, start_time=start_time, end_time=end_time)
        counts = counts.rename(
            columns={
                "start": "TimeStamp"}).set_index(
            ["TimeStamp"])
        counts.index = pd.to_datetime(counts.index)
        ret = {"counts": counts}
        if search:
            logging.info(
                "Search tweets for query {} from {} to {}".format(
                    query, start_time, end_time
                )
            )
            entities = ["annotations", "hashtags", "mentions", "urls"]
            public_metrics = [
                "retweet_count",
                "reply_count",
                "like_count",
                "like_count",
            ]

            def get_meta_info(tweet):
                for key in entities:
                    if key in tweet.get("entities", set()):
                        tweet[key] = len(tweet["entities"][key])
                for key in public_metrics:
                    if key in tweet.get("public_metrics", set()):
                        tweet[key] = tweet["public_metrics"][key]
                return tweet

            tweets = self.search_tweets(
                query, start_time=start_time, end_time=end_time,
                func=get_meta_info)
            tweets = tweets.rename(
                columns={"created_at": "TimeStamp"}).set_index(
                ["TimeStamp"])
            for key in entities + public_metrics:
                if key not in tweets.columns:
                    tweets[key] = 0
                else:
                    tweets[key].fillna(0, inplace=True)
            try:
                del tweets["entities"], tweets["public_metrics"]
            except Exception:
                pass
            tweets.index = pd.to_datetime(tweets.index)
            ret["tweets"] = tweets
        return ret

    def analyze_entity(self, url: str, start_time: str = "",
                       end_time: str = ""):
        query = 'url:"{}"'.format(url)
        tweets = self.search_tweets(
            query, start_time=start_time, end_time=end_time)
        counts = self.count_tweets(
            query, start_time=start_time, end_time=end_time)
        return {"tweets": tweets, "counts": counts}

    @staticmethod
    def visualize_count(counts: pd.DataFrame):
        counts = counts[["end", "tweet_count"]].rename(
            columns={"end": "TimeStamp", "tweet_count": "TweetCounts"}
        )
        for index in [counts.index, counts.index[::-1]]:
            for idx in index:
                if counts.loc[idx]["TweetCounts"] == 0:
                    counts.drop(idx, inplace=True)
                else:
                    break
        chart = (
            alt.Chart(counts)
            .mark_line(point=alt.OverlayMarkDef(color="red"))
            .encode(
                x=alt.X("TimeStamp:T"),
                y=alt.Y("TweetCounts:Q"),
                tooltip=["TimeStamp:T", "TweetCounts:Q"],
            )
        )
        return chart

    def get_cand_tweets_count(
            self, df_cand: pd.DataFrame, start_time: str, end_time: str,
            status=Dict[str, Any]) -> pd.DataFrame:
        status["df_count"] = df_count = list()
        for i, (name, twitter) in enumerate(df_cand["Twitter"].iteritems()):
            status["i"] = i
            status["username"] = username = twitter.split("/")[-1]
            query = f"(from:{username}) OR (@{username})"
            try:
                df_cand_count = self.count_tweets(
                    query=query, start_time=start_time, end_time=end_time
                )
                df_cand_count["Name"] = name
                df_count.append(df_cand_count)
            except Exception:
                logging.error(f"{name} {traceback.format_exc()}")
        status["df_count"] = df_count = pd.concat(df_count).set_index(["Name"])
        df_count["Date"] = pd.DatetimeIndex(
            pd.to_datetime(df_count["start"])).strftime("%Y%m%d")
        return df_count.drop(["start", "end"], axis=1)

    def get_cand_count_from_users(
            self, df_cand: pd.DataFrame, users: pd.Series, start_time: str,
            end_time: str, status=Dict[str, Any]) -> pd.DataFrame:
        status["df_count"] = df_count = list()
        for i, (name, twitter) in enumerate(df_cand["Twitter"].iteritems()):
            status["i"] = i
            status["username"] = username = twitter.split("/")[-1]
            query = ""
            for j, sus_user in enumerate(users):
                status["j"] = j
                if query == "":
                    query = f"@{username} ((from:{sus_user})"
                next_query = query + f" OR (from:{sus_user})"
                if len(next_query) > 950:
                    try:
                        df_cand_count = self.count_tweets(
                            query=f"{query})", start_time=start_time,
                            end_time=end_time)
                        df_cand_count["Name"] = name
                        df_cand_count["SusName"] = sus_user
                        df_count.append(df_cand_count)
                    except Exception:
                        logging.error(f"{name} {traceback.format_exc()}")
                    query = f"((@{username})) ((from:{sus_user})"
                else:
                    query = next_query
            try:
                df_cand_count = self.count_tweets(
                    query=f"{query})", start_time=start_time,
                    end_time=end_time)
                df_cand_count["Name"] = name
                df_cand_count["SusName"] = sus_user
                df_count.append(df_cand_count)
            except Exception:
                logging.error(f"{name} {traceback.format_exc()}")

        status["df_count"] = df_count = pd.concat(df_count).set_index(["Name"])
        df_count["Date"] = pd.DatetimeIndex(
            pd.to_datetime(df_count["start"])).strftime("%Y%m%d")
        return df_count.drop(["start", "end"], axis=1)

    def get_cand_count_from_domain(
            self, df_cand: pd.DataFrame, domains: pd.Series, start_time: str,
            end_time: str, status=Dict[str, Any]) -> pd.DataFrame:
        status["df_count"] = df_count = list()
        for i, (name, twitter) in enumerate(df_cand["Twitter"].iteritems()):
            status["i"] = i
            status["username"] = username = twitter.split("/")[-1]
            query = ""
            for j, domain in enumerate(domains):
                status["j"] = j
                if query == "":
                    query = f"@{username} ((url:{domain})"
                next_query = query + f" OR (url:{domain})"
                if len(next_query) > 950:
                    try:
                        df_domain_count = self.count_tweets(
                            query=f"{query})", start_time=start_time,
                            end_time=end_time)
                        df_domain_count["Name"] = name
                        df_domain_count["Domain"] = domain
                        df_count.append(df_domain_count)
                    except Exception:
                        logging.error(f"{name} {traceback.format_exc()}")
                    query = f"((@{username})) ((url:{domain})"
                else:
                    query = next_query
            try:
                df_domain_count = self.count_tweets(
                    query=f"{query})", start_time=start_time,
                    end_time=end_time)
                df_domain_count["Name"] = name
                df_domain_count["Domain"] = domain
                df_count.append(df_domain_count)
            except Exception:
                logging.error(f"{name} {traceback.format_exc()}")

        status["df_count"] = df_count = pd.concat(df_count).set_index(["Name"])
        df_count["Date"] = pd.DatetimeIndex(
            pd.to_datetime(df_count["start"])).strftime("%Y%m%d")
        return df_count.drop(["start", "end"], axis=1)

    def search_cand_tweets(
            self, df_cand: pd.DataFrame, start_time: str, end_time: str,
            status=Dict[str, Any]) -> pd.DataFrame:
        status["res"] = res = list()
        for i, (name, row) in enumerate(df_cand.iterrows()):
            status["user"] = user = row.Twitter.split("/")[-1]
            status["i"] = i
            try:
                tweets = self.get_user_tweets(
                    user=user, start_time=start_time, end_time=end_time)
                if not tweets.empty:
                    tweets["Name"] = user
                    res.append(tweets)
            except Exception:
                logging.error(
                    f"Error when name={name}, {traceback.format_exc()}")
        return pd.concat(status["res"])

    def get_user_tweets(self, user: str, start_time: str,
                        end_time: str) -> pd.DataFrame:
        tweets = list()
        try:
            query = f"(from:{user}) OR (@{user})"
            tweets.append(
                self.search_tweets(
                    query=query,
                    start_time=start_time,
                    end_time=end_time))
        except Exception:
            logging.error(f"Error when user={user}, {traceback.format_exc()}")
        return pd.concat(tweets)

    @classmethod
    def clean_tweets(cls,
                     df_tweets: pd.DataFrame, df_cand: pd.DataFrame,
                     checker: DomainChecker) -> pd.DataFrame:
        df_tweets.columns = [col.upper()[0] + col[1:]
                             for col in df_tweets.columns]
        df_tweets["Content"] = cls.parse_tweet(df_tweets["Text"])
        df_tweets["Entities"] = df_tweets["Entities"].fillna("").astype(str)
        df_tweets["Credibility"] = df_tweets["Entities"].apply(checker.check)
        df_tweets["Date"] = pd.DatetimeIndex(
            df_tweets["Created_at"]).strftime("%Y%m%d")
        name_dt = {
            row["Twitter"].split("/")[-1]: row["Name"] for i,
            row in df_cand.iterrows()}
        df_tweets["Name"] = df_tweets["Name"].map(name_dt)
        for col in ["Id", "Author_id"]:
            df_tweets[col] = df_tweets[col].astype(int)
        return df_tweets.set_index(["Id"]).sort_values(["Date", "Name"])


if __name__ == "__main__":
    with open("TwitterAPI.json", "r") as f:
        api_config = json.loads(f.read())["backup"]
        api = TwitterFactCheck(api_config)
    df_cand = pd.read_csv(
        "Data/Candidates/Candidates.csv",
        sep="\t").dropna(
        subset=["Position"])
    df_sus_users = pd.read_csv("Data/Network/NetworkUsers.csv", sep="\t")
    end_date = datetime.now() - timedelta(days=1)
    start_date = end_date - timedelta(days=1)
