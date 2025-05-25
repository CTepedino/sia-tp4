import csv


def read_europe_dataset(path):
    countries = []

    with open(path, "r") as f:
        reader = csv.reader(f)
        for line in reader:
            countries = {
                "country": line[0],
                "area": int(line[1]),
                "gdp": int(line[2]),
                "inflation": float(line[3]),
                "life.expect": float(line[4]),
                "military": float(line[5]),
                "pop.growth": float(line[6]),
                "unemployment": float(line[7])
            }

    return countries

