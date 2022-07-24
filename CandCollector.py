import json
import logging
import requests
import traceback
from typing import Dict, Iterable, List

from bs4 import BeautifulSoup
import pandas as pd

from TwitterFactCheck import TwitterFactCheck

logger = logging.getLogger("CandCollector")


class CandCollector(object):
    url = "https://ballotpedia.org"
    positions = [
        "U.S. Senate",
        "Governor",
        "State Senate",
        "House of Representatives",
        "Supreme Court",
        "Appeals",
        "Governor",
        "U.S. House",
    ]

    def __init__(self, api, df_cand, states: Iterable[str]):
        self.api = api
        self.df_cand = df_cand
        self.states = states

    @classmethod
    def scrap_twitter_account(cls, href: str) -> Dict[str, str]:
        """Scrap the twitter account of the candidate from its personal webpage from ballotpedia"""
        res = requests.get(href)
        page = BeautifulSoup(res.text)
        position = ""
        for p in page.find_all("p"):
            text = p.text
            if "running for" in text:
                position = "Secretary of State"
                for pos in cls.positions:
                    if pos in text:
                        position = pos
                        break
        contacts = page.findAll(
            "div", {"class": "widget-row value-only white"})
        info = {"position": position}
        for contact in contacts:
            if "republican" in contact.text.lower():
                info["Party"] = "Republican"
            elif "democratic" in contact.text.lower():
                info["Party"] = "Democratic"
            elif "green" in contact.text.lower():
                info["Party"] = "Green"
            if "twitter" in contact.text.lower():
                info["Twitter"] = contact.find("a")["href"]
        return info

    def find_profile(self, cand_page, state) -> List[Dict[str, str]]:
        cands_info = list()
        for row in cand_page.find_all("a"):
            name = row.text.strip()
            if name in self.df_cand["Name"] or "https" in name or name == "":
                continue
            try:
                href = row["href"]
                label = "_".join(row.text.split(" ")).lower()
                if href.lower().endswith(label) and href.lower().startswith(
                    "https://ballotpedia.org/"
                ):
                    info = self.scrap_twitter_account(href)
                    info["Name"], info["href"], info["State"] = name, href, state
                    cands_info.append(info)
            except Exception:
                logger.error(f"{state} {name}: {traceback.format_exc()}")
        return cands_info

    def find_candidate(self, state: str, position: str) -> pd.DataFrame:
        """Fetch all the candidate information in a given state"""
        cand_info = list()
        for state in self.states:
            state_page = requests.get(
                "{}/{}_elections,_2022".format(self.url, state))
            state_page = BeautifulSoup(state_page.text)
            for pos in self.positions:
                cand_page = state_page.find("a", text=pos)["href"]
                cand_page = requests.get("{}{}".format(self.url, cand_page))
                cand_page = BeautifulSoup(cand_page.text)
                cand_info.extend(self.find_profile(cand_page, state))

        return pd.DataFrame(cand_info)


if __name__ == "__main__":
    with open("TwitterApiKey.json") as f:
        keys = json.loads(f.read())
        api = TwitterFactCheck(f["main"])
    df_cand = pd.read_csv(
        "Data/Candidates/Candidates.csv",
        sep="\t",
        index_col="Name")
    df_cand = df_cand.dropna(subset="Position")
    with open("Data/GeoInfo.json", "r") as f:
        states = list(json.loads(f.read())["States"].values())
    collector = CandCollector(api, df_cand, states)
