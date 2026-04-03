def resumo_grupo(df):
    return {
        "sentimento_medio": df["sentimento"].mean(),
        "negativos_%": (df["sentimento"] < -0.1).mean() * 100
    }