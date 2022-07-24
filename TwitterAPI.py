from datetime import datetime, timedelta
import json
import logging
from multiprocessing.dummy import Process
import requests
import time
from typing import Callable, Union, Dict, Iterable, Set

import pandas as pd

logger = logging.getLogger("TwitterAPI Logger")


class TwitterAPI(object):
    url = "https://api.twitter.com/2/"
    default_tweet_params = {
        "tweet.fields": {
            "author_id",
            "entities",
            "created_at",
            "source",
            "public_metrics",
        },
        "user.fields": {
            "created_at",
            "entities",
            "location",
            "url",
            "username"},
        "max_results": 500,
    }
    default_user_id_params = {
        "user.fields": {
            "created_at",
            "location",
            "name",
            "username",
            "verified",
            "public_metrics",
        }
    }

    def __init__(self, api_tokens: Union[Dict[str, str], str]):
        """A crawler to get Tweets from V2 API

        Args:
            api_token (Union[Dict[str, str], str]): User API token
        """
        if isinstance(api_tokens, str):
            with open(api_tokens, "r") as f:
                self._api_tokens = json.loads(f.read())
        else:
            self._api_tokens = api_tokens

    @staticmethod
    def convert_time(start_time: str = "", end_time: str = ""):
        """Conver the time input into the valid type for twitter API"""
        if not end_time:
            end_time = datetime.now()
        if not start_time:
            start_time = end_time - timedelta(days=7)
        if isinstance(start_time, str):
            start_time = datetime.strptime(start_time, "%Y%m%d")
        if isinstance(end_time, str):
            end_time = datetime.strptime(end_time, "%Y%m%d")
        return {
            "start_time": start_time.isoformat() + "Z",
            "end_time": end_time.isoformat() + "Z",
        }

    def get_api_result(self, api: str, kwargs: Dict[str, object] = dict()):
        """Use requests library to get information from tweets

        Args:
            api(str): The specific api
            kwargs(Dict[str, object]): Other paremeters used for query

        Returns:
            (Object): The query result of tweets
        """
        headers = {
            "Authorization": "Bearer {}".format(
                self._api_tokens["token"])}
        while True:
            response = requests.request(
                "GET", self.url + api, headers=headers, params=kwargs
            )
            logger.debug(response.url)
            if response.status_code == 429:
                if json.loads(response.text)["title"] == "UsageCapExceeded":
                    logging.critical("Tweet Usage Cap Exceeded")
                    raise Exception(response.status_code, response.text)
                time.sleep(3)
                continue
            elif response.status_code != 200:
                raise Exception(response.status_code, response.text)
            break

        return response.json()

    def search_tweet(self, twitter_ids: Iterable[Union[int, str]]) -> Dict:
        headers = {
            "Authorization": "Bearer {}".format(
                self._api_tokens["token"])}
        if isinstance(twitter_ids, (int, str)):
            kwargs = {"ids": [str(twitter_ids)]}
        else:
            kwargs = {"ids": ",".join([str(twitter_id)
                                      for twitter_id in twitter_ids])}
        response = requests.request(
            "GET", self.url + "tweets", headers=headers, params=kwargs
        )
        logger.debug(response.url)
        return response.json()

    def search_tweets(
        self,
        query: str,
        *,
        params: Dict = dict(),
        start_time: str = "",
        end_time: str = "",
        func: Callable = None
    ) -> Iterable[Dict]:
        """Search tweets with certain period

        Args:
            query(str): The query information related to twitter
            params(Dict): Other paremeters used for query
            start_time(pd.Timestamp): The start time of the search
            end_time(pd.Timestamp): The end time of the search
        Returns:
            (Dict): The query result of tweets
        """
        kwargs = self.convert_time(start_time, end_time)
        for key, val in self.default_tweet_params.items():
            kwargs[key] = val if key not in params else params[key]
            if hasattr(kwargs[key], "__iter__") and not isinstance(
                    kwargs[key], str):
                kwargs[key] = ",".join(kwargs[key])
        kwargs["query"] = query
        tweets, results, i = list(), list(), 0

        def fetch_tweets():
            while True:
                query_result = self.get_api_result("tweets/search/all", kwargs)
                tweets.extend(query_result.get("data", list()))
                if "next_token" not in query_result["meta"].keys():
                    break
                kwargs["next_token"] = query_result["meta"]["next_token"]
                time.sleep(1.5)

        fetcher = Process(target=fetch_tweets)
        fetcher.start()
        while fetcher.is_alive() or i < len(tweets):
            if i < len(tweets):
                results.append(func(tweets[i]) if func else tweets[i])
                i += 1
        logger.info("Query {} tweets in total".format(len(results)))
        if not results:
            return pd.DataFrame()
        return pd.DataFrame(results)

    def count_tweets(
            self, query: str, start_time: str = "", end_time: str = "",
            freq: str = "day") -> Iterable[int]:
        """Count tweets with certain period

            Args:
                query(str): The query information related to twitter
                start_time(pd.Timestamp): The start time of the search
                end_time(pd.Timestamp): The end time of the search
                freq(str): The period for counting. Can be day/hour/minute
            Returns:
                (Dict): The query result of tweets
        """
        kwargs = self.convert_time(start_time, end_time)
        kwargs["query"], kwargs["granularity"] = query, freq
        results = list()
        while True:
            query_result = self.get_api_result("tweets/counts/all", kwargs)
            results.extend(query_result.get("data", list()))
            if "next_token" not in query_result["meta"].keys():
                break
            kwargs["next_token"] = query_result["meta"]["next_token"]
            time.sleep(1.5)
        logger.debug("Query {} tweets volume completed".format(len(results)))
        return pd.DataFrame(results).sort_values(["end"])

    def search_user_tweets(self, user_name: str, *, params: Dict = dict()):
        """Search tweets by user

        Args:
            user_name(str): The poster of tweets
            params(Dict): Other paremeters used for query
        Returns:
            (Dict): The query result of tweets
        """
        kwargs = {"username": user_name}
        results = list()
        for key in {"tweet.fields", "user.fields"}:
            kwargs[key] = params.get(key, self.default_tweet_params[key])

        while True:
            query_result = self.get_api_result("/users/by/username", kwargs)
            results.extend(query_result.get("data", list()))
            if "next_token" not in query_result["meta"].keys():
                break
            kwargs["next_token"] = query_result["meta"]["next_token"]
            time.sleep(1)
        return pd.DataFrame(results)

    def search_user(
        self,
        user_id: Union[int, str],
        fields: Set[str] = {
            "followers_count",
            "tweet_count",
            "username",
            "verified",
            "created_at",
            "location",
        },
        *,
        params: Dict = dict()
    ):
        """Search user by user_id

        Args:
            user_id (Union[int, str]): The id of the user
            params(Dict): Other paremeters used for query
        Returns:
            (Dict): The query result of user
        """
        kwargs, res = dict(), dict()
        for key, val in self.default_user_id_params.items():
            kwargs[key] = val if key not in params else params[key]
            if hasattr(kwargs[key], "__iter__") and not isinstance(
                    kwargs[key], str):
                kwargs[key] = ",".join(kwargs[key])

        headers = {
            "Authorization": "Bearer {}".format(
                self._api_tokens["token"])}
        response = requests.request(
            "GET", self.url + "users/{}".format(user_id),
            headers=headers, params=kwargs)
        logger.debug(response.url)

        def parse(info):
            for key, val in info.items():
                if isinstance(val, dict):
                    parse(val)
                elif key in fields:
                    res[key] = val
        return res

    @staticmethod
    def parse_tweet(series: pd.Series) -> pd.Series:
        series = series.str.replace(r"(@[\w|\d]+|\#[\w|\d]+|https\S+)", " ")
        for s in [r"\s{2,}", r"RT:\s?", r"^s+\$", r"\\u", r"[^\s\w,!?]"]:
            series = series.str.replace(s, "")
        return series.str.replace(r"\s+", " ")
