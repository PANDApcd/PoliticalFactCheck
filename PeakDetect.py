from itertools import product
import warnings

import pandas as pd
from pandas.core.common import SettingWithCopyWarning
import numpy as np
from scipy.signal import find_peaks
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import altair as alt
from torch import dropout

warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)


class PeakDetector(object):
    df_count_columns = [
        'Name',
        'Date',
        'TweetCount',
        'SusUserCount',
        'SusDomainCount',
        'MonthTweetCount',
        'TweetPeakIQR',
        'SusUserPeakIQR',
        'SusDomainPeakIQR']

    def __init__(self, df_cand: pd.DataFrame, df_sus_users: pd.DataFrame,
                 df_new_tweets: pd.DataFrame, df_count: pd.DataFrame = None):
        self.df_sus_users = df_sus_users
        self.df_cand = df_cand
        start_date = df_new_tweets["Date"].min()
        end_date = df_new_tweets["Date"].max()
        if df_count is None:
            df_count = pd.DataFrame(columns=self.df_count_columns)
        else:
            start_date = min(start_date, df_count["Date"].min())
            end_date = max(end_date, df_count["Date"].max())
        dates = pd.date_range(str(start_date), str(
            end_date)).strftime("%Y%m%d").astype(int)
        df_index = pd.DataFrame(index=product(df_cand["Name"], dates))
        df_count = df_count.set_index(["Name", "Date"])
        df_index = df_index[~df_index.index.isin(df_count.index)]
        df_count = pd.concat([df_count, df_index])
        self.df_count = df_count.reset_index()
        self.df_count["Month"] = df_new_tweets["Date"].astype(int) // 100
        if df_new_tweets.index.name != "Id":
            df_new_tweets = df_new_tweets.set_index(["Id"])
        df_new_tweets = df_new_tweets[~df_new_tweets.index.duplicated()]
        df_new_tweets["Month"] = df_new_tweets["Date"].astype(int) // 100
        self.df_new_tweets = df_new_tweets[df_new_tweets["Name"].isin(
            df_cand["Name"])].dropna(subset=["Content"])
        self.df_sus_user_tweets = df_new_tweets[df_new_tweets["Author_id"].isin(
            df_sus_users["User_id"])]
        self.df_sus_domain_tweets = df_new_tweets[df_new_tweets["Credibility"] == 0]

    def __call__(self):
        self.add_sus_tweets_count()
        self.add_monthly_count()
        self.df_count = self.generate_peak(df_count)
        df_count = self.df_count
        for col in df_count.columns:
            if not col.endswith("IQR") and col != "Name":
                df_count[col] = df_count[col].astype(int)
        return df_count

    def add_sus_tweets_count(self):
        for tweet_type, df_tweets in zip(["Tweet", "SusUser", "SusDomain"], [
                                         self.df_new_tweets, self.df_sus_user_tweets, self.df_sus_domain_tweets]):
            df_tweets = self.count_tweets(df_tweets)
            self.df_count = self.df_count.set_index(["Name", "Date"])
            df_tweets = df_tweets.set_index(["Name", "Date"])
            self.df_count["Text"] = df_tweets["Text"]
            self.df_count[f"{tweet_type}Count"] = np.where(
                self.df_count["Text"].isna(),
                self.df_count[f"{tweet_type}Count"],
                self.df_count["Text"])
            self.df_count = self.df_count.reset_index().drop(["Text"], axis=1)

    @classmethod
    def add_monthly_count(cls, df_count: pd.DataFrame) -> pd.DataFrame:
        if "MonthTweetCount" in df_count.columns:
            df_count = df_count.drop(["MonthTweetCount"], axis=1) 
        df_count["Month"] = df_count["Date"] // 100
        df_month_count = df_count.groupby(["Name", "Month"])["TweetCount"].sum()
        df_month_count.name = "MonthTweetCount"
        df_count = df_count.set_index(["Name", "Month"])
        df_count = df_count.merge(
            df_month_count, how="outer", left_index=True, right_index=True)
        return df_count.reset_index()

    def count_tweets(self, df_tweets: pd.DataFrame):
        df_tweets = df_tweets[~df_tweets.index.duplicated()]
        df_tweets["Date"] = df_tweets["Date"].astype(int)
        df_tweets = df_tweets.groupby(["Name", "Date"])[
            "Text"].count().reset_index()
        df_tweets = df_tweets[df_tweets["Name"].isin(self.df_cand["Name"])]
        return df_tweets

    @classmethod
    def generate_peak(cls, df_count: pd.DataFrame, iqrs=[1.5, 3, 4]) -> pd.DataFrame:
        df_ct_list = list()
        for name in df_count["Name"].drop_duplicates():
            df_cand_ct = df_count[df_count["Name"] == name]
            df_cand_ct = df_cand_ct.reset_index(
                drop=True)
            for count_type in ["Tweet", "SusUser", "SusDomain"]:
                for iqr in iqrs:
                    for i in cls.detect_peak(
                        df_cand_ct
                        [f"{count_type}Count"],
                            iqr=iqr):
                        df_cand_ct.at[i, f"{count_type}PeakIQR"] = iqr
            df_ct_list.append(df_cand_ct)
        df_count = pd.concat(df_ct_list).fillna(0).reset_index(drop=True)
        return cls.add_monthly_count(df_count)

    @staticmethod
    def detect_peak(counts: pd.Series, iqr: float = 1.5):
        if counts.empty:
            return list()
        prominence = iqr * (np.percentile(counts,
                            75) - np.percentile(counts, 25))
        peaks_indexes, _ = find_peaks(counts, prominence=prominence)
        return peaks_indexes

    @staticmethod
    def plot_peak(df_count: pd.DataFrame, field: str):
        df_count["Counts"] = df_count[field] / df_count[field].max()
        df_count = df_count[["Counts", "Date"]]
        df_count["Date"] = pd.to_datetime(df_count["Date"].astype(str))
        chart = alt.Chart(df_count).mark_line().encode(
            y=alt.Y("Counts:Q"),
            x=alt.X("Date:T"),
            tooltip=["Counts:Q", "Date:T"]
        )
        return chart

    @staticmethod
    def expand_peaks(df_count: pd.DataFrame, slide: int=1):
        df_count["Peak"] = (df_count["TweetPeakIQR"] > 0) | (df_count["SusUserPeakIQR"] > 0) | (df_count["SusDomainPeakIQR"] > 0)
        df_counts = list(df_count.groupby(["Name"]))
        for _, df in df_counts:
            for i in range(1, slide + 1):
                df["Peak"] = df["Peak"] | df["Peak"].shift(i).fillna(False) | df["Peak"].shift(-i).fillna(False)
        return pd.concat([df for _, df in df_counts])

    @staticmethod
    def get_metrics(
            df_count: pd.DataFrame, df_pf: pd.DataFrame) -> pd.DataFrame:
        metrics = ["f1", "precision", "recall", "accuracy"]
        metric_dt = dict()
        df_mf = df_pf[~df_pf["Rate"].str.contains(
            "True")].dropna(subset="Name")
        df_merged = df_mf.set_index(
            ["Date", "Name"]).merge(
            df_count.set_index(["Date", "Name"]),
            how="right", left_index=True, right_index=True)
        for method, iqr in product(
            ["Tweet", "SusUser", "SusDomain"],
                [1.5, 3, 4]):
            metric_dt[(method, iqr)] = list()
            label = ~df_merged["Rate"].isna()
            pred = df_merged[f"{method}PeakIQR"] >= iqr
            for metric in metrics:
                func = eval(f"{metric}_score")
                metric_dt[(method, iqr)].append(func(label, pred))
        return pd.DataFrame(metric_dt, index=metrics).T


if __name__ == "__main__":
    df_cand = pd.read_csv(
        "Data/Candidates/Candidates.csv",
        sep="\t").dropna(
        subset=["Position"])
    df_sus_users = pd.read_csv("Data/Network/NetworkUsers.csv", sep="\t")
    df_tweets = pd.read_csv("Data/Candidates/NewTweets.csv", sep="\t")
    df_count = pd.read_csv(
        "Data/Candidates/CandTweetsCount.csv",
        sep="\t").drop_duplicates(
        [
            "Name",
            "Date"]).reset_index(
                drop=True)
    pg = PeakDetector(df_cand, df_sus_users, df_tweets, df_count)
    pg.add_monthly_count()
    pg.add_sus_tweets_count()
    pg.generate_peak()
