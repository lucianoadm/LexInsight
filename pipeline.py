import pandas as pd
from pathlib import Path
import re
import spacy
from textblob import TextBlob

from assets.civil_lexicon import TERMOS_CIVIS_NEUTROS

# ======================================================
# CAMINHOS E LÉXICOS
# ======================================================
BASE_DIR = Path(__file__).resolve().parent

LEXICO_NEG = pd.read_csv(BASE_DIR / "assets" / "lexico_negativo.csv")
LEXICO_POS = pd.read_csv(BASE_DIR / "assets" / "lexico_positivo.csv")

LEXICO_NEG["termo"] = LEXICO_NEG["termo"].str.lower()
LEXICO_POS["termo"] = LEXICO_POS["termo"].str.lower()

LEXICO_NEG_DICT = dict(zip(LEXICO_NEG["termo"], LEXICO_NEG["peso"]))
LEXICO_POS_DICT = dict(zip(LEXICO_POS["termo"], LEXICO_POS["peso"]))

# ======================================================
# SENTIMENTO HÍBRIDO (MODELO + LÉXICO)
# ======================================================
def analisar_sentimento_hibrido(texto: str) -> float:
    texto = texto.lower()

    try:
        score_modelo = TextBlob(texto).sentiment.polarity
    except Exception:
        score_modelo = 0.0

    ajuste = 0.0
    hits = 0

    for termo, peso in LEXICO_NEG_DICT.items():
        if termo in texto:
            ajuste += peso
            hits += 1

    for termo, peso in LEXICO_POS_DICT.items():
        if termo in texto:
            ajuste += peso
            hits += 1

    if hits > 0:
        ajuste /= hits

    if abs(score_modelo) < 0.05 and hits > 0:
        peso_lexico = 0.7
        peso_modelo = 0.3
    else:
        peso_lexico = 0.4
        peso_modelo = 0.6

    score_final = (score_modelo * peso_modelo) + (ajuste * peso_lexico)
    return max(min(score_final, 1.0), -1.0)

# ======================================================
# NLP — LAZY LOADING (CRÍTICO PARA O CLOUD)
# ======================================================
_NLP = None
_STOP_WORDS = None

def get_nlp():
    global _NLP, _STOP_WORDS

    if _NLP is None:
        _NLP = spacy.load(
            "pt_core_news_sm",
            disable=["ner", "parser", "textcat"]
        )

        # stopwords nativas do spaCy (Cloud‑safe)
        _STOP_WORDS = _NLP.Defaults.stop_words.copy()

        stop_words_juridicas = {
            "autos", "processo", "deferido", "indeferido",
            "vossa", "excelencia", "data", "venia"
        }
        _STOP_WORDS.update(stop_words_juridicas)

    return _NLP, _STOP_WORDS

# ======================================================
# ETAPA 1: INPUT
# ======================================================
def carregar_textos(lista_textos):
    return pd.DataFrame(lista_textos, columns=["texto"])

# ======================================================
# ETAPA 2: LIMPEZA
# ======================================================
_regex_num = re.compile(r"\d+")
_regex_pont = re.compile(r"[^\w\s]")

def limpar_texto(texto: str) -> str:
    texto = texto.lower()
    texto = _regex_num.sub("", texto)
    texto = _regex_pont.sub("", texto)
    return texto

# ======================================================
# ETAPA 3: NLP (BATCH)
# ======================================================
def processar_textos(textos):
    nlp, stop_words = get_nlp()

    docs = nlp.pipe(textos, batch_size=50)
    tokens_processados = []

    for doc in docs:
        tokens = [
            token.lemma_
            for token in doc
            if token.is_alpha and token.lemma_ not in stop_words
        ]
        tokens_processados.append(tokens)

    return tokens_processados

# ======================================================
# PIPELINE FINAL
# ======================================================
def pipeline(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["limpo"] = df["texto"].apply(limpar_texto)
    df["tokens"] = processar_textos(df["limpo"].tolist())
    df["sentimento"] = df["limpo"].apply(analisar_sentimento_hibrido)

    return df
