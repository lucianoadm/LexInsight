import pandas as pd
from pathlib import Path
import re
import spacy
from spacy.lang.pt.stop_words import STOP_WORDS
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
# LAZY LOAD DO SPACY (CRÍTICO PARA DEPLOY)
# ======================================================
_nlp = None

def get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load(
            "pt_core_news_sm",
            disable=["ner", "parser", "textcat"]
        )
    return _nlp

# ======================================================
# STOPWORDS
# ======================================================
stop_words = set(STOP_WORDS)

stop_words_juridicas = {
    "autos", "processo", "deferido", "indeferido",
    "vossa", "excelencia", "data", "venia"
}

stop_words.update(stop_words_juridicas)

# ======================================================
# SENTIMENTO BASEADO EM LÉXICO (ROBUSTO)
# ======================================================
def analisar_sentimento(texto: str) -> float:
    texto = texto.lower()

    score = 0.0
    hits = 0

    for termo, peso in LEXICO_NEG_DICT.items():
        if termo in texto:
            score += peso
            hits += 1

    for termo, peso in LEXICO_POS_DICT.items():
        if termo in texto:
            score += peso
            hits += 1

    if hits > 0:
        score = score / hits

    return max(min(score, 1.0), -1.0)

# ======================================================
# INPUT
# ======================================================
def carregar_textos(lista_textos):
    return pd.DataFrame(lista_textos, columns=["texto"])

# ======================================================
# LIMPEZA
# ======================================================
_regex_num = re.compile(r"\d+")
_regex_pont = re.compile(r"[^\w\s]")

def limpar_texto(texto: str) -> str:
    texto = texto.lower()
    texto = _regex_num.sub("", texto)
    texto = _regex_pont.sub("", texto)
    return texto

# ======================================================
# NLP (BATCH)
# ======================================================
def processar_textos(textos):
    nlp = get_nlp()
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
    df["sentimento"] = df["limpo"].apply(analisar_sentimento)

    return df
