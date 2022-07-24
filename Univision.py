import requests
from bs4 import BeautifulSoup
from multiprocessing.dummy import Pool
import pandas as pd
import traceback
from datetime import datetime


class Univision(object):
    false_labels = {
        "falso",
        "enganoso",
        "falta-contexto",
        "no-hay-evidencia",
        "manipulado",
    }

    def __init__(self, html: BeautifulSoup,
                 threads: int = 5, timeout: int = 3):
        self.fact_links = dict()
        self.pool = Pool(threads)
        self.timeout = timeout
        for fact_link in html.find_all("div"):
            if set(fact_link.attrs["class"]) & self.false_labels:
                url = fact_link.find("a")["href"]
                if (
                    "rusia" in url
                    or "putin" in url
                    or "ucran" in url
                    or "zelensky" in url
                ):
                    continue
                self.fact_links[url] = {
                    "Rate": set(fact_link.attrs["class"]) & self.false_labels,
                    "Headline": fact_link.find("h2").find("a").text,
                }
                self.pool.apply_async(
                    self.get_fact_detail, kwds={
                        "fact_url": url})

    def get_fact_detail(self, fact_url: str):
        try:
            fact_response = requests.get(fact_url, timeout=self.timeout)
            fact_soup = BeautifulSoup(fact_response.text, "html.parser")
            article = fact_soup.find("article")
            urls = set()
            divs = article.find_all("div", {"class": "_2oxpe"})
            for div in divs:
                for url in div.find_all("a"):
                    urls.add(url["href"])
            self.fact_links[fact_url]["EntityURL"] = sorted(urls)
            self.fact_links[fact_url]["TimeStamp"] = datetime.strptime(
                fact_soup.find("span", {"class": "SCSiX"}).find(
                    "meta")["content"][:19],
                "%Y-%m-%dT%H:%M:%S",
            )

        except Exception:
            print(traceback.format_exc())

    def get_results(self):
        df = pd.DataFrame(self.fact_links).T
        df.index.name = "CheckURL"
        return df
