import requests
import re
from bs4 import BeautifulSoup
import pandas as pd
from multiprocessing.dummy import Pool
import traceback
import logging
import time
from typing import Callable
import os

# logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("PolitiFact Logger")


class PolitiFact(object):
    API = "https://www.politifact.com"

    def __init__(self, threads: int = 5, timeout: int = 3):
        self.timeout = timeout
        self.pool = Pool(threads)
        self.threads = threads
        self.work_count = 0
        self.results = list()
        self.complete_work = list()
        self.reinit()

    def reinit(self):
        if self.is_running:
            self.pool.terminate()
            self.pool = Pool(self.threads)
            self.complete_work = list()
            self.results = list()
            self.work_count = 0

    @property
    def is_running(self):
        return self.work_count != len(self.complete_work)

    def get_results(self):
        df = pd.DataFrame(self.results).set_index(["CheckURL"])
        df["Date"] = pd.to_datetime(
            df["CheckTime"]).astype(str).str.replace(
            "-", "")
        return df

    def run_task(self, func: Callable, kwds):
        kwds = kwds["kwds"]
        try:
            func(**kwds)
        except Exception:
            logger.warning(
                "Task error when kwds={}: {}".format(
                    kwds, traceback.format_exc())
            )
        self.complete_work.append(0)

    def apply_async(self, func: Callable, **kwds):
        self.work_count += 1
        self.pool.apply_async(self.run_task, kwds={"func": func, "kwds": kwds})

    def get_fact_detail(self, fact_url: str):
        """Scrap the fact webpage and genereate the data of each fact

        Args:
            fact_url(str): The web url of each fact
        """
        ret = dict()
        fact_response = requests.get(fact_url, timeout=self.timeout)
        fact_soup = BeautifulSoup(fact_response.text, "html.parser")
        ret["CheckURL"] = fact_url
        ret["Poster"] = fact_soup.find(
            "a", {"class": "m-statement__name"})["title"]
        ret["Setting"] = fact_soup.find(
            "div", {"class": "m-statement__desc"}
        ).text.replace("\n", "")
        ret["Statement"] = fact_soup.find(
            "div", {"class": "m-statement__quote"}
        ).text.replace("\n", "")
        ret["CheckTime"] = fact_soup.find(
            "span", {"class": "m-author__date"}
        ).text.replace("\n", "")
        ret["Tags"] = ";".join(
            [
                tag.find("span").text
                for tag in fact_soup.find_all("a", {"class": "c-tag"})
            ]
        )
        ret["Rate"] = None
        scripts = fact_soup.find_all("script", {"type": "text/javascript"})
        for script in scripts:
            try:
                ret["Rate"] = re.search(
                    r"'Truth-O-Meter': '[^']*'", script.text
                ).group()[18:-1]
                break
            except Exception:
                pass
        source_soup = BeautifulSoup(fact_response.text, "html.parser")
        sources = source_soup.find(
            "article", {"class": "m-superbox__content"}
        ).find_all("p")
        ret["Sources"] = ""
        for source in sources[:-1]:
            source_title = source.text
            if source_title == "\xa0":
                continue
            source_url = ""
            try:
                source_url = source.find("a")["href"]
            except Exception:
                pass
            if ret["Sources"]:
                ret["Sources"] += "<source-sep>"
            ret["Sources"] += "({})[{}]".format(source_title, source_url)
        self.results.append(pd.Series(ret))

    def get_page_facts(self, page_id: int) -> pd.DataFrame:
        """Get fact data from politifact.com on certain page

        Args:
            page_id(int): The index of pages where we fetch data
        """
        fact_list = requests.get(
            "{}/factchecks/list/".format(self.API),
            params={"page": page_id},
            timeout=self.timeout,
        )
        fact_list_soup = BeautifulSoup(fact_list.text, "html.parser")
        for div in fact_list_soup.find_all(
                "div", {"class": "m-statement__content"}):
            fact_url = "{}{}".format(self.API, div.find("a")["href"])
            if (
                "putin" in fact_url
                or "ukrain" in fact_url
                or "russia" in fact_url
                or "zelensky" in fact_url
            ):
                continue
            try:  # get the response of each fact
                self.get_fact_detail(fact_url)
            except Exception:
                logger.warning(
                    "Page {} {} fact error: {}".format(
                        page_id, fact_url, traceback.format_exc()
                    )
                )
            else:
                logger.info(
                    "Page {} {} fact complete".format(
                        page_id, fact_url))

    def get_facts(self, page_num: int = 15):
        """Get fact data from politifact.com

        Args:
            page_num(int): The number of pages where we fetch data, each page will use one thread
        """
        self.reinit()
        for page_id in range(page_num):
            self.apply_async(
                self.get_page_facts, kwds={
                    "page_id": page_id + 1})

    def get_recent_facts(self, topic: str):
        """Get recent facts result from politicfact.com

        Args:
            topic(str): The topic or key words for search
        """
        self.reinit()
        res = requests.get(
            self.API + "/search/factcheck", params={"q": topic}, timeout=1
        )
        recent_fact_soup = BeautifulSoup(res.text, "html.parser")
        recent_facts = recent_fact_soup.find_all(
            "div", {"class": "c-textgroup__title"})
        for fact in recent_facts:
            fact_url = self.API + fact.find("a")["href"]
            self.apply_async(self.get_fact_detail, kwds={"fact_url": fact_url})

    def get_name(self, row, df_cand):
        '''get the politican's name of each fact check'''
        for tag in row.Tags.split(";"):
            if tag in df_cand.Name.tolist():
                return tag
        for name in df_cand.Name:
            if name in row.Statement or name in row.Poster:
                return name
        return ""


if __name__ == "__main__":
    api = PolitiFact(10)
    api.get_facts(20)
    while api.is_running:
        time.sleep(1)
        pass
    df = api.get_results()
    path = "Data/PolitiFact.csv"
    df_cand = pd.read_csv(
        "Data/Candidates/Candidates.csv",
        sep="\t").dropna(
        subset="Position")

    if not os.path.isfile(path):
        df["iteration"] = 0
        df.to_csv("Data/PolitiFact.csv", index_label="CheckURL", sep="\t")
    else:
        exist_df = pd.read_csv(
            "Data/PolitiFact.csv",
            index_col="CheckURL",
            sep="\t")
        df["iteration"] = exist_df["iteration"].max() + 1
        df = pd.concat([exist_df, df])
        df[~df.index.duplicated(keep="first")].to_csv(
            "Data/PolitiFact.csv", index_label="CheckURL", sep="\t"
        )
