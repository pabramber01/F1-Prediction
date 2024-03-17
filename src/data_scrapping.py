from bs4 import BeautifulSoup
from urllib.request import Request, urlopen
from urllib.parse import quote
import unicodedata as uni
import re


import pandas as pd

from pathlib import Path


def weather():
    races_df = pd.read_csv("./src/data/races.csv")

    weather_df = races_df.iloc[:, [1, 2]].copy()

    info = []

    for url in races_df.url:
        html = BeautifulSoup(urlopen(Request(url)).read(), features="lxml")
        try:
            info.append(
                html.find("th", string=re.compile("weather", re.IGNORECASE))
                .next_sibling.text.strip()
                .replace("\n", " ")
            )
        except:
            info.append("not found")

    weather_df["weather"] = info

    weather_dict = {
        "weatherWarm": [
            "clear",
            "warm",
            "hot",
            "sunny",
            "fine",
            "mild",
        ],
        "weatherCold": [
            "cold",
            "fresh",
            "chilly",
            "cool",
        ],
        "weatherDry": [
            "dry",
        ],
        "weatherWet": [
            "shower",
            "wet",
            "rain",
            "damp",
            "thunderstorm",
        ],
        "weatherCloudy": [
            "overcast",
            "cloud",
            "grey",
        ],
    }

    for col in weather_dict:
        weather_df[col] = weather_df["weather"].map(
            lambda s: 1 if any(w in s.lower() for w in weather_dict[col]) else 0
        )

    weather_df.to_csv("./src/data/weather.csv", index=False)


def circuits_plus():
    url = "https://en.wikipedia.org/wiki/List_of_Formula_One_circuits"

    circuits_df = pd.read_csv("./src/data/circuits.csv")

    res = []

    html = BeautifulSoup(urlopen(Request(url)).read(), features="lxml")
    table = html.find_all("table")[2]
    for t_row in table.find_all("tr")[1:]:
        name = t_row.contents[1].text.strip().replace(" *", "").replace(" ", "_")
        type = t_row.contents[5].text.strip()
        direction = t_row.contents[7].text.strip()
        length = t_row.contents[13].text.strip()
        res.append((name, type, direction, length))

    circuits_plus_df = pd.DataFrame(
        res,
        columns=[
            "circuitName",
            "circuitType",
            "circuitDirection",
            "circuitLength",
        ],
    )

    length_formatter = lambda l: float(l.split("km")[0].strip(" ").strip("\xa0"))

    circuits_plus_df["circuitLength"] = circuits_plus_df["circuitLength"].apply(
        length_formatter
    )

    a = lambda s: "".join(c for c in uni.normalize("NFD", s) if uni.category(c) != "Mn")
    name_formatter = lambda n: next(
        iter(
            circuits_df.loc[
                (
                    circuits_df["circuitRef"].apply(
                        lambda r: bool(re.search(a(r), a(n), re.I))
                    )
                )
                | (
                    circuits_df["name"].apply(
                        lambda m: bool(re.match(a(m.replace(" ", "_")), a(n), re.I))
                    )
                )
                | (
                    circuits_df["circuitRef"]
                    .str.fullmatch("portimao")
                    .apply(lambda b: b and n == "Algarve_International_Circuit")
                )
                | (
                    circuits_df["circuitRef"]
                    .str.fullmatch("imola")
                    .apply(
                        lambda b: b
                        and n == "Autodromo_Internazionale_Enzo_e_Dino_Ferrari"
                    )
                )
                | (
                    circuits_df["circuitRef"]
                    .str.fullmatch("yeongam")
                    .apply(lambda b: b and n == "Korea_International_Circuit")
                )
                | (
                    circuits_df["circuitRef"]
                    .str.fullmatch("losail")
                    .apply(lambda b: b and n == "Lusail_International_Circuit")
                ),
                "circuitRef",
            ]
        ),
        "None",
    )

    circuits_plus_df["circuitName"] = circuits_plus_df["circuitName"].apply(
        name_formatter
    )

    circuits_plus_df.rename(columns={"circuitName": "circuitRef"}, inplace=True)

    circuits_plus_df.to_csv("./src/data/circuits_plus.csv", index=False)


def circuits_plusplus():
    races_df = pd.read_csv("./src/data/races.csv")
    circuits_df = pd.read_csv("./src/data/circuits.csv")

    races_df = races_df.merge(
        circuits_df, how="left", on="circuitId", suffixes=("", "_")
    )

    circuits_plusplus_df = races_df.loc[:, ["circuitRef"]].copy()

    laps = []
    dist = []

    for url in races_df.url:
        html = BeautifulSoup(urlopen(Request(url)).read(), features="lxml")
        try:
            td = (
                html.find("th", string=re.compile("distance", re.IGNORECASE))
                .next_sibling.text.strip()
                .replace("\n", " ")
                .split(" ")
            )
            l = int(td[0].split("[")[0])
            d = float(td[2].split("[")[0].split("km")[0])
            laps.append(l)
            dist.append(d)
        except:
            laps.append("not found")
            dist.append("not found")

    circuits_plusplus_df["circuitLaps"] = laps
    circuits_plusplus_df["circuitDist"] = dist

    circuits_plusplus_df.drop_duplicates("circuitRef", keep="last").to_csv(
        "./src/data/circuits_plusplus.csv", index=False
    )


def driver_ratings_ea():
    urls_by_year = {
        "2023": "https://www.the-race.com/formula-1/new-f1-games-driver-ratings-show-ea-has-learned-some-lessons/",
        "2022": "https://www.racefans.net/2022/06/24/formula-1-driver-ratings-for-new-f1-22-game-revealed/",
        "2021": "https://www.the-race.com/formula-1/f1-2021-driver-ratings-unveiled-as-verstappen-equals-hamilton/",
        "2020": "https://www.gptoday.net/en/news/f1/255983/all-driver-ratings-from-f1-2020-game-revealed",
    }

    res = []

    for year, url in urls_by_year.items():
        html = BeautifulSoup(urlopen(Request(url)).read(), features="lxml")
        t_rows = html.find_all("tr")[1:]
        for t_row in t_rows:
            driver_data = [year]
            for t_col in t_row.children:
                data = t_col.text.strip()
                if data:
                    if data.isnumeric():
                        driver_data.append(float(data))
                    else:
                        driver_data.append("_".join(data.split(" ")).lower())
            res.append(tuple(driver_data))

    pd.DataFrame(
        res,
        columns=[
            "season",
            "driverName",
            "driverExp",
            "driverRac",
            "driverAwa",
            "driverPac",
            "driverOvr",
        ],
    ).to_csv("./src/data/driver_ratings_ea.csv", index=False)


if __name__ == "__main__":
    if not Path("./src/data/weather.csv").is_file():
        weather()
    if not Path("./src/data/circuits_plus.csv").is_file():
        circuits_plus()
    if not Path("./src/data/circuits_plusplus.csv").is_file():
        circuits_plusplus()
    if not Path("./src/data/driver_ratings_ea.csv").is_file():
        driver_ratings_ea()
