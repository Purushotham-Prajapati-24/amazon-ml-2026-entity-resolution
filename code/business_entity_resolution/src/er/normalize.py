"""Text normalisation for business names and addresses.

Country-agnostic by design: the test set contains France, which never appears in
training, so nothing here branches on the country label. All scripts are mapped to
Latin with ``anyascii`` (ISC licence) so Indic names/addresses become comparable with
the Latin-only Source 1.

Every rule here comes from inspecting the provided data (frequent address components,
noise patterns listed in the brief); no external data is looked up.
"""
import re
from functools import lru_cache

from anyascii import anyascii

# --------------------------------------------------------------------------- tokens
_APOS = re.compile(r"['’`´]")
_ZERO_IN_WORD = re.compile(r"(?<=[a-z])0(?=[a-z])")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_DIGITS = re.compile(r"\d+")
_URL_HINT = re.compile(r"(www\.|https?:|\.(com|in|net|org|co|fr|info|biz|us)\b)", re.I)
# Case-mangled UTF-8 mojibake: "â\x80\x93" / "Â\x80\x93" are the bytes E2 80 xx of a
# punctuation char (en dash, quotes, ...).
_MOJIBAKE = re.compile(r"[âÂ]\x80([\x80-\xbf])")


def _fix_mojibake(m: re.Match) -> str:
    return bytes([0xE2, 0x80, ord(m.group(1))]).decode("utf-8", "replace")


def tokens(text: str) -> list[str]:
    """Transliterate, lowercase, fix 0-for-o typos, drop punctuation; return tokens."""
    if not text:
        return []
    if "\x80" in text:
        text = _MOJIBAKE.sub(_fix_mojibake, text)
    s = anyascii(text).lower().replace("&", " and ")
    s = _APOS.sub("", s)
    s = _ZERO_IN_WORD.sub("o", s)
    return _NON_ALNUM.sub(" ", s).split()


# --------------------------------------------------------------------------- names
# Legal forms and fillers removed from the comparison key (US / India / France).
LEGAL_WORDS = frozenset(
    "llc inc lnc incorporated corp corporation co company ltd limited pvt private plc llp lp pllc pc "
    "sarl sas sasu sa sci eurl snc cie ei scop selarl the and of et".split()
)
URL_WORDS = frozenset("www http https com in net org co fr info biz us".split())
NAME_ABBR = {
    "ctr": "center", "cntr": "center", "cter": "center", "centre": "center",
    "svc": "service", "svcs": "service", "services": "service",
    "intl": "international", "mfg": "manufacturing", "assoc": "association",
    "assn": "association", "univ": "university", "hosp": "hospital",
    "mgmt": "management", "technologies": "tech", "technology": "tech",
    "bros": "brothers", "dept": "department", "natl": "national",
    "enterprises": "enterprise", "industries": "industry", "solutions": "solution",
    "associates": "associate", "systems": "system", "traders": "trader",
}


def merge_initials(toks: list[str]) -> list[str]:
    """Join runs of single letters: dotted abbreviations ("S.A.S.", "L.L.C.", "E.U.R.L.")
    tokenise to one letter per token and would otherwise never match "SAS" / "LLC"."""
    out, run = [], []
    for t in toks + [""]:
        if len(t) == 1 and t.isalpha():
            run.append(t)
            continue
        out.extend(["".join(run)] if len(run) >= 2 else run)
        run = []
        if t:
            out.append(t)
    return out


def normalize_name(raw: str) -> dict:
    toks = merge_initials(tokens(raw))
    is_url = bool(_URL_HINT.search(raw))
    if is_url:
        toks = [t for t in toks if t not in URL_WORDS]
    core = [NAME_ABBR.get(t, t) for t in toks if t not in LEGAL_WORDS]
    return {
        "name_norm": " ".join(toks),
        "name_core": " ".join(core),               # legal-free, original order
        "name_key": " ".join(sorted(set(core))),   # order-insensitive exact key
        "name_url": is_url,
    }


# --------------------------------------------------------------------------- states
# canonical code -> aliases (raw spellings; normalised through ``tokens`` at import).
# Codes shared across countries (e.g. GA = Georgia / Goa) are harmless because
# candidates are only ever compared within one country.
_STATE_ALIASES = {
    # United States
    "al": ["alabama"], "ak": ["alaska"], "az": ["arizona"], "ar": ["arkansas", "arunachal pradesh"],
    "ca": ["california"], "co": ["colorado"], "ct": ["connecticut"], "de": ["delaware"],
    "dc": ["district of columbia"], "fl": ["florida"], "ga": ["georgia", "goa"], "hi": ["hawaii"],
    "id": ["idaho"], "il": ["illinois"], "in": ["indiana"], "ia": ["iowa"], "ks": ["kansas"],
    "ky": ["kentucky"], "la": ["louisiana"], "me": ["maine"], "md": ["maryland"],
    "ma": ["massachusetts"], "mi": ["michigan"], "mn": ["minnesota"], "ms": ["mississippi"],
    "mo": ["missouri"], "mt": ["montana"], "ne": ["nebraska"], "nv": ["nevada"],
    "nh": ["new hampshire"], "nj": ["new jersey"], "nm": ["new mexico"], "ny": ["new york"],
    "nc": ["north carolina"], "nd": ["north dakota"], "oh": ["ohio"], "ok": ["oklahoma"],
    "or": ["oregon"], "pa": ["pennsylvania"], "ri": ["rhode island"], "sc": ["south carolina"],
    "sd": ["south dakota"], "tn": ["tennessee", "tamil nadu", "தமிழ்நாடு"], "tx": ["texas"],
    "ut": ["utah"], "vt": ["vermont"], "va": ["virginia"], "wa": ["washington"],
    "wv": ["west virginia"], "wi": ["wisconsin"], "wy": ["wyoming"], "pr": ["puerto rico"],
    # India (S3 uses these codes; S2/S3 also carry native-script names)
    "mh": ["maharashtra", "महाराष्ट्र"], "dl": ["delhi", "nct of delhi", "दिल्ली"],
    "up": ["uttar pradesh", "उत्तर प्रदेश"], "ka": ["karnataka", "ಕರ್ನಾಟಕ"],
    "wb": ["west bengal", "পশ্চিমবঙ্গ"], "gj": ["gujarat", "ગુજરાત"],
    "tg": ["telangana", "ts", "తెలంగాణ"], "hr": ["haryana", "हरियाणा"],
    "rj": ["rajasthan", "राजस्थान"], "kl": ["kerala", "keralam", "കേരളം"], "br": ["bihar", "बिहार"],
    "mp": ["madhya pradesh", "मध्य प्रदेश"], "ap": ["andhra pradesh", "ఆంధ్ర ప్రదేశ్"],
    "pb": ["punjab", "ਪੰਜਾਬ"], "od": ["odisha", "orissa", "ଓଡ଼ିଶା"], "as": ["assam", "অসম"],
    "jh": ["jharkhand", "झारखंड"], "cg": ["chhattisgarh", "छत्तीसगढ़"],
    "uk": ["uttarakhand", "uttaranchal", "उत्तराखंड"], "hp": ["himachal pradesh"],
    "jk": ["jammu and kashmir", "jammu kashmir"], "ch": ["chandigarh"], "py": ["puducherry", "pondicherry"],
    # France: regions, with the departments seen in the data folded into their region
    "fr_hdf": ["hauts de france", "nord", "pas de calais", "somme", "oise", "aisne"],
    "fr_naq": ["nouvelle aquitaine", "gironde", "landes", "dordogne", "charente", "charente maritime"],
    "fr_pdl": ["pays de la loire", "loire atlantique", "vendee", "maine et loire", "sarthe", "mayenne"],
    "fr_idf": ["ile de france", "paris"], "fr_ara": ["auvergne rhone alpes"],
    "fr_bfc": ["bourgogne franche comte"], "fr_bre": ["bretagne"], "fr_cvl": ["centre val de loire"],
    "fr_ges": ["grand est"], "fr_nor": ["normandie"], "fr_occ": ["occitanie"],
    "fr_pac": ["provence alpes cote dazur", "provence alpes cote d azur"], "fr_cor": ["corse"],
}


@lru_cache(maxsize=1)
def _state_lookup() -> dict:
    lookup = {}
    for code, aliases in _STATE_ALIASES.items():
        lookup[code] = code
        for a in aliases:
            lookup[" ".join(tokens(a))] = code
    return lookup


# --------------------------------------------------------------------------- addresses
NULL_COMPONENTS = frozenset({"null", "n a", "na", "none", "nil", "nan"})
ADDR_ABBR = {
    # street types (US / France share "st" for street/saint)
    "street": "st", "str": "st", "saint": "st", "avenue": "ave", "av": "ave", "road": "rd",
    "drive": "dr", "boulevard": "blvd", "bd": "blvd", "boul": "blvd", "bvd": "blvd",
    "highway": "hwy", "hiway": "hwy", "lane": "ln", "court": "ct", "place": "pl",
    "parkway": "pkwy", "pky": "pkwy", "circle": "cir", "terrace": "ter", "terr": "ter",
    "square": "sq", "expressway": "expy", "freeway": "fwy", "trail": "trl", "point": "pt",
    "mount": "mt", "fort": "ft", "suite": "ste", "sainte": "ste", "apartment": "apt",
    "building": "bldg", "bldng": "bldg", "floor": "fl", "flr": "fl",
    "north": "n", "south": "s", "east": "e", "west": "w", "northeast": "ne",
    "northwest": "nw", "southeast": "se", "southwest": "sw",
    # India
    "near": "nr", "opposite": "opp", "sector": "sec", "ngr": "nagar", "col": "colony",
    "bengaluru": "bangalore", "calcutta": "kolkata", "gurugram": "gurgaon", "bombay": "mumbai",
    "madras": "chennai",
    # France
    "r": "rue", "all": "allee", "chem": "chemin", "imp": "impasse", "rte": "route",
    "faubourg": "fbg", "residence": "res", "batiment": "bat", "rpt": "rondpoint",
}
ADDR_DROP = frozenset("no nos num number de du des la le les l d of the".split())


def normalize_address(raw: str) -> dict:
    """Split on commas, pull state/region components into their own field, canonicalise the rest."""
    states, toks = set(), []
    lookup = _state_lookup()
    for comp in raw.split(","):
        ct = tokens(comp)
        if not ct:
            continue
        joined = " ".join(ct)
        if joined in NULL_COMPONENTS:
            continue
        code = lookup.get(joined)
        if code is not None:
            states.add(code)
            continue
        for t in ct:
            if t in ADDR_DROP:
                continue
            toks.append(ADDR_ABBR.get(t, t))
    numbers = sorted({str(int(m)) for m in _DIGITS.findall(" ".join(toks))})
    return {
        "addr_norm": " ".join(toks),
        "states": " ".join(sorted(states)),
        "numbers": " ".join(numbers),
    }


def normalize_record(name: str, address: str) -> dict:
    return {**normalize_name(name), **normalize_address(address)}
