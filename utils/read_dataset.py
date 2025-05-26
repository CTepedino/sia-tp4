import csv


def read_europe_dataset(path):
    countries = []

    with open(path, "r") as f:
        f.readline()
        reader = csv.reader(f)
        for line in reader:
            countries.append({
                "country": line[0],
                "data": {
                    "area": int(line[1]),
                    "gdp": int(line[2]),
                    "inflation": float(line[3]),
                    "life.expect": float(line[4]),
                    "military": float(line[5]),
                    "pop.growth": float(line[6]),
                    "unemployment": float(line[7])
                }
            })

    return countries

def read_europe_dataset_as_matrix(path):
    countries = []
    data = []
    with open(path, "r") as f:
        f.readline()
        reader = csv.reader(f)
        for line in reader:
            countries.append(line[0])
            data.append([
                int(line[1]),
                int(line[2]),
                float(line[3]),
                float(line[4]),
                float(line[5]),
                float(line[6]),
                float(line[7])
            ])

    return countries, data