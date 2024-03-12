from bs4 import BeautifulSoup
from urllib.request import Request, urlopen

import pandas as pd

from pathlib import Path


def ea_driver_ratings():
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
    if not Path("./src/data/driver_ratings_ea.csv").is_file():
        ea_driver_ratings()
