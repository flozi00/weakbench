"""Builds the private set tasks_novel.GEO_PLACES from GeoNames (CC BY 4.0, https://www.geonames.org/):
download cities15000.zip, countryInfo.txt and admin1CodesASCII.txt into geo_dl/, unzip, run `python build_geo_places.py`,
upload wbdata/sets/tasks_novel.GEO_PLACES.py to the dataset repo and bump common.DATA_REV."""
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
GEO_DE = {"Cairo": "Kairo", "Moscow": "Moskau", "Beijing": "Peking", "Tokyo": "Tokio", "Mexico City": "Mexiko-Stadt",
          "Cape Town": "Kapstadt", "Tehran": "Teheran", "Riyadh": "Riad", "Kyiv": "Kiew", "Bucharest": "Bukarest", "Warsaw": "Warschau",
          "Vienna": "Wien", "Saint Petersburg": "Sankt Petersburg", "Addis Ababa": "Addis Abeba", "Rome": "Rom", "Milan": "Mailand",
          "Munich": "München", "Cologne": "Köln", "Lisbon": "Lissabon", "Athens": "Athen", "Brussels": "Brüssel", "Prague": "Prag",
          "Copenhagen": "Kopenhagen", "Seoul": "Seoul", "New Delhi": "Neu-Delhi", "Bogotá": "Bogotá", "Havana": "Havanna"}
import math
def km(a, b):
    return 6371 * math.acos(min(1, math.sin(math.radians(a[0])) * math.sin(math.radians(b[0])) +
                                math.cos(math.radians(a[0])) * math.cos(math.radians(b[0])) * math.cos(math.radians(a[1] - b[1]))))
# well-known big cities: 1.5M+ with many names (fame), and a capital or a top-3 city of its country; no duplicates
feat = {}
for ln in open("geo_dl/cities15000.txt", encoding="utf-8"):
    f = ln.split("\t")
    feat[(round(float(f[4]), 3), round(float(f[5]), 3))] = f[7]
cand = sorted([x for x in rows if x[3] >= 1_500_000 and len(x[4]) >= 20], key=lambda x: -x[3])
rank = {}
big, seen = [], set()
for x in cand:
    rank[x[2]] = rank.get(x[2], 0) + 1
    if (rank[x[2]] <= 3 or feat.get(x[:2]) == "PPLC") and x[4][0] not in seen and x[4][0] not in ("Brooklyn", "Queens"):
        seen.add(x[4][0])
        big.append(x)
primaries = {x[4][0].lower() for x in rows if x[3] >= 300_000}
large = [x for x in rows if x[3] >= 300_000]
pool_city = []
for la, lo, c, p, nm in big:
    names = [nm[0]] + [n for n in nm[1:] if n.lower() not in primaries]  # no other city as an alias (Giza is not Cairo)
    near = [y[4][0] for y in large if y[4][0] != nm[0] and km((la, lo), y[:2]) <= 30]  # accepted for geo_coords only
    pool_city.append((la, lo, c, p, names, near, GEO_DE.get(nm[0], nm[0])))
# towns whose name is unambiguous: not used by any other place (any name or alternate name in cities15000), no region,
# state or country name (admin1 codes, country names, Gold Coast), no city district (PPLX), and well known enough to have
# 8+ alternate names (drops database-only records); a short deny list for the rest
DENY = {"paraná", "surrey", "delta", "north shore", "kimberley", "sunshine coast", "gold coast", "springs", "nigel", "mestre",
        "archway", "cajamarca", "posadas", "ocaña", "crato", "queluz", "coimbra", "münster", "batu", "sutton", "logan city",
        "montreuil", "lampa", "simmering", "ajax", "the bronx", "buda", "pest"}
namecount = {}
recs = []
for ln in open("geo_dl/cities15000.txt", encoding="utf-8"):
    f = ln.rstrip("\n").split("\t")
    alts = [a for a in f[3].split(",") if a]
    for nm in {f[1].lower(), f[2].lower(), *(a.lower() for a in alts)}:
        namecount[nm] = namecount.get(nm, 0) + 1
    recs.append((f[1], f[8], int(f[14] or 0), f[7], len(alts), f[2].lower()))  # the native Latin name is shown
admin = {ln.split("\t")[1].lower() for ln in open("geo_dl/admin1CodesASCII.txt", encoding="utf-8")} | \
        {ln.split("\t")[2].lower() for ln in open("geo_dl/admin1CodesASCII.txt", encoding="utf-8")}
countryish = {x.lower() for en, de in C.values() for x in en + de} | {"gold coast", "ceylon", "persia", "siam", "burma"}
unique_towns = []
by = {}
for ascii_name, cc, pop, fcode, nalt, name in recs:
    a = ascii_name.lower()
    if (cc in C and pop >= 100_000 and fcode != "PPLX" and nalt >= 8 and namecount.get(a) == 1 and namecount.get(name, 1) == 1
            and a not in admin and a not in countryish and a not in DENY and name not in DENY
            and re.fullmatch(r"[A-Za-zÀ-ÖØ-öø-ÿĀ-ž' -]{4,}", ascii_name)):
        by.setdefault(cc, []).append((ascii_name, cc, pop))
for cc in sorted(by):
    lst = sorted(by[cc])
    unique_towns += r.sample(lst, min(10, len(lst)))
# country facts from countryInfo.txt: area (km²), continent, land neighbours; the capital's names from its PPLC row
facts = {}
for ln in open("geo_dl/countryInfo.txt", encoding="utf-8"):
    if ln.startswith("#"):
        continue
    f = ln.rstrip("\n").split("\t")
    if f[0] in C:
        facts[f[0]] = {"area": float(f[6] or 0), "continent": f[8], "neighbours": [x for x in f[17].split(",") if x], "capital": [f[5]]}
for ln in open("geo_dl/cities15000.txt", encoding="utf-8"):
    f = ln.rstrip("\n").split("\t")
    if f[7] == "PPLC" and f[8] in facts:
        alts = [x for x in f[3].split(",") if re.fullmatch(r"[A-Za-zÀ-ÿ' .-]{3,}", x) and x.lower() not in COUNTRY_WORDS]
        facts[f[8]]["capital"] = sorted({f[1], f[2], *facts[f[8]]["capital"], *alts}, key=lambda x: x != f[1])
out = {"countries": C, "facts": facts, "decoys": DECOYS, "country_points": pool_country, "big_cities": pool_city, "towns": unique_towns}
s = repr(out)
open("wbdata/sets/tasks_novel.GEO_PLACES.py", "w", encoding="utf-8").write(s)
print(len(pool_country), len(pool_city), len(s))
