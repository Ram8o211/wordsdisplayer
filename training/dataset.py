"""Synthetic, weakly-supervised corpus of spoken-style sentences labelled with
visual scene concepts and affect (valence / arousal).

The corpus is built from hand-curated lexicons (English + Italian) and
sentence templates. It is deliberately simple so that the whole pipeline is
reproducible offline; `train.py --extra data.jsonl` lets you mix in a real
annotated corpus ({"text": ..., "concepts": {...}, "valence": v, "arousal": a}).
"""
import json
import random

from common import CONCEPTS

# concept -> (keywords, default valence, default arousal)
LEXICON = {
    "ocean": ("ocean sea seas waves wave tide surf sailing sail ship boat boats harbor coast "
              "coastline sailor voyage oceano mare onde nave barca porto marea", 0.3, 0.3),
    "beach": ("beach sand sandy shore seaside tropical palm palms island vacation holiday "
              "sunscreen spiaggia sabbia isola vacanza vacanze riva", 0.7, 0.3),
    "forest": ("forest forests woods woodland trees tree pine oak jungle leaves moss hiking "
               "trail wilderness foresta bosco alberi albero giungla foglie sentiero", 0.4, -0.1),
    "mountain": ("mountain mountains peak peaks summit alps hill hills cliff climbing climb "
                 "valley glacier ridge montagna montagne vetta collina valle scalata", 0.4, 0.2),
    "desert": ("desert dunes dune sahara cactus dry drought arid camel oasis heat scorching "
               "deserto duna dune cammello oasi siccita caldo", -0.1, 0.1),
    "city": ("city cities town street streets downtown traffic buildings building skyscraper "
             "skyline urban metro subway avenue neighborhood citta strada strade traffico "
             "palazzi grattacielo quartiere", 0.0, 0.5),
    "space": ("space universe galaxy galaxies planet planets cosmos astronaut rocket orbit "
              "mars moon nebula spaceship stars alien satellite spazio universo galassia "
              "pianeta pianeti razzo astronauta luna stelle", 0.3, 0.4),
    "night": ("night tonight midnight dark darkness evening moonlight nocturnal sleep dream "
              "dreams late notte buio sera mezzanotte sogno sogni dormire", -0.1, -0.4),
    "rain": ("rain raining rainy drizzle shower showers umbrella wet puddle puddles pouring "
             "pioggia piove piovere ombrello bagnato pozzanghera", -0.3, -0.2),
    "snow": ("snow snowing snowy winter frozen ice icy cold frost snowflake snowman ski "
             "skiing blizzard neve nevica inverno ghiaccio freddo gelo sciare", 0.2, -0.1),
    "fire": ("fire flames flame burning burn campfire fireplace blaze smoke wildfire candle "
             "ember heat fuoco fiamme fiamma bruciare falo candela fumo incendio", 0.0, 0.7),
    "sunset": ("sunset sunrise dawn dusk twilight golden horizon sundown daybreak orange "
               "tramonto alba crepuscolo orizzonte aurora", 0.6, -0.2),
    "river": ("river rivers stream creek waterfall lake lakes brook flowing canal pond bridge "
              "fiume torrente cascata lago ruscello ponte canale", 0.4, 0.0),
    "flowers": ("flower flowers garden gardens roses rose bloom blooming tulips daisy spring "
                "petals blossom meadow fiori fiore giardino rose primavera petali", 0.8, 0.1),
    "storm": ("storm storms thunder lightning hurricane tornado thunderstorm gale tempest "
              "wind windy cyclone tempesta temporale fulmine tuono uragano vento", -0.5, 0.9),
    "underwater": ("underwater diving diver scuba coral reef fish fishes dolphin dolphins whale "
                   "whales shark submarine deep jellyfish sott acqua subacqueo corallo pesci "
                   "delfino balena squalo sottomarino", 0.4, 0.1),
    "technology": ("technology computer computers software code coding robot robots ai "
                   "artificial intelligence internet digital data network algorithm chip "
                   "machine smartphone tecnologia computer robot intelligenza artificiale "
                   "rete dati algoritmo digitale", 0.2, 0.5),
    "music": ("music song songs singing sing concert band guitar piano drums melody rhythm "
              "dance dancing party festival dj musica canzone canzoni cantare concerto "
              "chitarra pianoforte ballare festa", 0.7, 0.8),
    "love": ("love loving lover heart hearts kiss romance romantic wedding darling sweetheart "
             "valentine hug together amore cuore bacio romantico matrimonio abbraccio "
             "innamorato tesoro", 0.9, 0.3),
    "clouds": ("cloud clouds cloudy sky skies fog foggy mist overcast airplane flying fly "
               "nuvole nuvola cielo nebbia volare aereo", 0.1, -0.1),
    "birds": ("bird birds eagle eagles seagull seagulls flock wings feathers sparrow owl "
              "parrot migration uccelli uccello aquila gabbiano ali piume gufo", 0.5, 0.3),
    "fields": ("field fields farm farmland countryside grass meadow wheat harvest village "
               "barn cows tractor prairie campagna campi campo fattoria grano erba villaggio",
               0.5, -0.2),
}

MOODS = {
    # word: (valence, arousal)
    "happy": (0.9, 0.5), "joy": (0.9, 0.6), "wonderful": (0.9, 0.4), "beautiful": (0.8, 0.2),
    "amazing": (0.9, 0.7), "peaceful": (0.7, -0.7), "calm": (0.5, -0.8), "quiet": (0.3, -0.7),
    "relaxing": (0.6, -0.7), "gentle": (0.5, -0.6), "exciting": (0.7, 0.9), "wild": (0.1, 0.9),
    "sad": (-0.8, -0.4), "lonely": (-0.7, -0.4), "tired": (-0.4, -0.7), "scary": (-0.7, 0.8),
    "terrible": (-0.9, 0.6), "angry": (-0.8, 0.9), "afraid": (-0.7, 0.7), "dangerous": (-0.6, 0.8),
    "furious": (-0.8, 1.0), "gloomy": (-0.6, -0.5), "cozy": (0.7, -0.5), "magical": (0.8, 0.4),
    "felice": (0.9, 0.5), "bellissimo": (0.8, 0.3), "tranquillo": (0.6, -0.7), "triste": (-0.8, -0.4),
    "paura": (-0.7, 0.8), "arrabbiato": (-0.8, 0.9), "calmo": (0.5, -0.8), "magico": (0.8, 0.4),
}

FILLERS = (
    "i think that we should talk about it . you know what i mean . so anyway let me tell you . "
    "yesterday my friend said something . it was really interesting . and then we decided to go . "
    "well actually i am not sure . what do you think about this . the meeting starts at nine . "
    "please send me the report . we need more time . that is a good question . "
    "allora ti dico una cosa . ieri il mio amico ha detto . non sono sicuro . cosa ne pensi ."
).split(" . ")

TEMPLATES = [
    "{p} {a}", "we went to the {a} and it was {m}", "i remember the {a} {p}",
    "look at the {a} over there", "there is a {m} {a} near the {b}",
    "imagine a {a} with {b}", "last summer we saw the {a}", "the {a} was so {m} today",
    "my favourite place is the {a}", "let me tell you about the {a} and the {b}",
    "{a} {b}", "{a}", "{m} {a}", "can you see the {a}", "i dream of {a} and {b} and {c}",
    "the story begins with a {a}", "she loves the {a}", "he talked about {a} all night long",
    "siamo andati al {a} ed era {m}", "ricordo il {a} e il {b}", "guarda il {a}",
    "a {m} {a} under the {b}", "suddenly the {a} appeared", "they walked through the {a}",
]
PREFIXES = ["", "well", "so", "and", "you know", "honestly", "oh", "then", "allora", "e poi"]


def _kw(concept, rng):
    return rng.choice(LEXICON[concept][0].split())


def make_sentence(rng: random.Random):
    """Return (text, {concept: weight}, valence, arousal)."""
    if rng.random() < 0.12:  # neutral talk, nothing to visualise
        return rng.choice(FILLERS), {}, 0.0, 0.0
    n = rng.choices([1, 2, 3], weights=[0.5, 0.35, 0.15])[0]
    cs = rng.sample(CONCEPTS, n)
    mood = rng.choice(list(MOODS)) if rng.random() < 0.6 else None
    t = rng.choice(TEMPLATES)
    slots = {"a": _kw(cs[0], rng), "b": _kw(cs[min(1, n - 1)], rng),
             "c": _kw(cs[min(2, n - 1)], rng), "m": mood or rng.choice(["nice", "big", "old", "real"]),
             "p": rng.choice(PREFIXES)}
    text = t.format(**slots).strip()
    used = {c for c, s in zip(cs, "abc") if "{" + s + "}" in t}
    if not used:
        used = {cs[0]}
    # optionally add the leftover concepts as an appended phrase
    for c in cs:
        if c not in used and rng.random() < 0.7:
            text += " " + rng.choice(["and", "with", "near", "e", "con"]) + " " + _kw(c, rng)
            used.add(c)
    if rng.random() < 0.25:
        text = rng.choice(FILLERS) + " " + text
    v = sum(LEXICON[c][1] for c in used) / len(used)
    a = sum(LEXICON[c][2] for c in used) / len(used)
    if mood and mood in text.split():
        mv, ma = MOODS[mood]
        v, a = 0.35 * v + 0.65 * mv, 0.35 * a + 0.65 * ma
    return text, {c: 1.0 for c in used}, v, a


def make_sample(rng: random.Random, extra=None):
    """A stream excerpt: 1-3 sentences; target emphasises the most recent one
    (soft targets) so that models learn to follow the *current* topic."""
    k = rng.choices([1, 2, 3], weights=[0.45, 0.4, 0.15])[0]
    sents = []
    for _ in range(k):
        if extra and rng.random() < 0.5:
            e = rng.choice(extra)
            sents.append((e["text"], e.get("concepts", {}), e.get("valence", 0.0), e.get("arousal", 0.0)))
        else:
            sents.append(make_sentence(rng))
    target = [0.0] * len(CONCEPTS)
    for age, (_, cs, _, _) in enumerate(reversed(sents)):
        w = [1.0, 0.35, 0.1][age]
        for c, cw in cs.items():
            i = CONCEPTS.index(c)
            target[i] = max(target[i], w * cw)
    _, _, v, a = sents[-1]
    text = " ".join(s[0] for s in sents)
    return text, target, v, a


def load_extra(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


if __name__ == "__main__":
    r = random.Random(0)
    for _ in range(10):
        print(make_sample(r))
