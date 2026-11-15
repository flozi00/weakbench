"""Builds the private set tasks_novel.GEO_PLACES from GeoNames (CC BY 4.0, https://www.geonames.org/):
download cities15000.zip into geo_dl/, unzip, run `python build_geo_places.py`, upload wbdata/sets/tasks_novel.GEO_PLACES.py
to the dataset repo and bump common.DATA_REV."""
import random, re, json, sys
# ISO code -> (English names, German names); only countries with stable, well-known names
C = {"DE": (["Germany"], ["Deutschland"]), "FR": (["France"], ["Frankreich"]), "IT": (["Italy"], ["Italien"]), "ES": (["Spain"], ["Spanien"]),
 "PT": (["Portugal"], ["Portugal"]), "GB": (["United Kingdom", "UK", "Britain", "Great Britain", "England", "Scotland", "Wales", "Northern Ireland"], ["Vereinigtes Königreich", "Großbritannien", "England", "Schottland", "Wales", "Nordirland"]),
 "IE": (["Ireland"], ["Irland"]), "NL": (["Netherlands", "Holland"], ["Niederlande", "Holland"]), "BE": (["Belgium"], ["Belgien"]),
 "CH": (["Switzerland"], ["Schweiz"]), "AT": (["Austria"], ["Österreich"]), "PL": (["Poland"], ["Polen"]), "CZ": (["Czech Republic", "Czechia"], ["Tschechien", "Tschechische Republik", "Tschechei"]),
 "SE": (["Sweden"], ["Schweden"]), "NO": (["Norway"], ["Norwegen"]), "FI": (["Finland"], ["Finnland"]), "DK": (["Denmark"], ["Dänemark"]),
 "GR": (["Greece"], ["Griechenland"]), "TR": (["Turkey", "Türkiye"], ["Türkei"]), "HU": (["Hungary"], ["Ungarn"]), "RO": (["Romania"], ["Rumänien"]),
 "UA": (["Ukraine"], ["Ukraine"]), "US": (["United States", "USA", "U.S.", "U.S.A.", "United States of America"], ["Vereinigte Staaten", "USA", "U.S.A."]),
 "CA": (["Canada"], ["Kanada"]), "MX": (["Mexico"], ["Mexiko"]), "BR": (["Brazil"], ["Brasilien"]), "AR": (["Argentina"], ["Argentinien"]),
 "CL": (["Chile"], ["Chile"]), "PE": (["Peru"], ["Peru"]), "CO": (["Colombia"], ["Kolumbien"]), "EG": (["Egypt"], ["Ägypten"]),
 "MA": (["Morocco"], ["Marokko"]), "NG": (["Nigeria"], ["Nigeria"]), "KE": (["Kenya"], ["Kenia"]), "ZA": (["South Africa"], ["Südafrika"]),
 "ET": (["Ethiopia"], ["Äthiopien"]), "IN": (["India"], ["Indien"]), "PK": (["Pakistan"], ["Pakistan"]), "CN": (["China"], ["China"]),
 "JP": (["Japan"], ["Japan"]), "KR": (["South Korea", "Korea", "Republic of Korea"], ["Südkorea", "Korea"]), "ID": (["Indonesia"], ["Indonesien"]),
 "TH": (["Thailand"], ["Thailand"]), "VN": (["Vietnam", "Viet Nam"], ["Vietnam"]), "PH": (["Philippines"], ["Philippinen"]),
 "AU": (["Australia"], ["Australien"]), "NZ": (["New Zealand"], ["Neuseeland"]), "IR": (["Iran"], ["Iran"]), "SA": (["Saudi Arabia"], ["Saudi-Arabien"]),
 "IL": (["Israel"], ["Israel"]), "RU": (["Russia", "Russian Federation"], ["Russland"]), "KZ": (["Kazakhstan"], ["Kasachstan"]), "IS": (["Iceland"], ["Island"])}
COUNTRY_WORDS = {x.lower() for en, de in C.values() for x in en + de}  # "Mexico" is no alias of Mexico City
# names that are not countries of the set but contain one ("North Korea", "South America"): matched, never gold
DECOYS = {"KP": (["North Korea", "DPRK"], ["Nordkorea"]), "XA": (["South America", "Central America", "Latin America", "North America"],
          ["Südamerika", "Mittelamerika", "Lateinamerika", "Nordamerika"])}
rows = []
for ln in open("geo_dl/cities15000.txt", encoding="utf-8"):
    f = ln.rstrip("\n").split("\t")
    cc, pop = f[8], int(f[14] or 0)
    if cc not in C or (cc == "UA" and f[10] in ("05", "08", "11", "14", "26")):  # occupied oblasts: no disputed gold
        continue
    alt = [a for a in f[3].split(",") if re.fullmatch(r"[A-Za-zÀ-ÿ' .-]{4,}", a) and a.lower() not in COUNTRY_WORDS]
    names = [f[1]] + ([f[2]] if f[2] != f[1] else []) + sorted(set(alt) - {f[1], f[2]})  # the GeoNames name first
    rows.append((round(float(f[4]), 3), round(float(f[5]), 3), cc, pop, names))
r = random.Random("geo_build")
towns = {}
for row in rows:
    towns.setdefault(row[2], []).append(row)
pool_country = []
for cc, lst in sorted(towns.items()):
    lst = sorted(lst, key=lambda x: (x[0], x[1]))
    pool_country += [(la, lo, c, p) for la, lo, c, p, _ in r.sample(lst, min(8, len(lst)))]
big = sorted([x for x in rows if x[3] >= 1_500_000], key=lambda x: -x[3])
import math
def km(a, b):
    return 6371 * math.acos(min(1, math.sin(math.radians(a[0])) * math.sin(math.radians(b[0])) +
                                math.cos(math.radians(a[0])) * math.cos(math.radians(b[0])) * math.cos(math.radians(a[1] - b[1]))))
large = [x for x in rows if x[3] >= 300_000]
pool_city = []
for la, lo, c, p, nm in big:  # a neighbouring big city within 30 km counts too (Soweto -> Johannesburg)
    near = [y for y in large if y[4][0] != nm[0] and km((la, lo), y[:2]) <= 30]
    pool_city.append((la, lo, c, p, nm[:1] + [n for n in nm[1:]] + [n for y in near for n in y[4][:2]]))
out = {"countries": C, "decoys": DECOYS, "country_points": pool_country, "big_cities": pool_city}
s = repr(out)
open("wbdata/sets/tasks_novel.GEO_PLACES.py", "w", encoding="utf-8").write(s)
print(len(pool_country), len(pool_city), len(s))
