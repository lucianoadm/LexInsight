import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Imports pesados com cache
from wordcloud import WordCloud
from io import BytesIO

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# Pipeline (importar aqui, mas funções pesadas com cache)
from pipeline import carregar_textos, pipeline

# ======================================================
# CONFIGURAÇÃO
# ======================================================
st.set_page_config(
    page_title="LexInsight",
    layout="wide",
    page_icon="⚖️",
    initial_sidebar_state="expanded"
)

st.title("⚖️ LexInsight — Análise de Sentimento Textual")

# ======================================================
# DASHBOARD COMPARATIVO
# ======================================================
st.subheader("📊 Comparação entre Grupos")

col1, col2 = st.columns(2)
with col1:
    texto_a = st.text_area("Texto - Grupo A", height=140)
with col2:
    texto_b = st.text_area("Texto - Grupo B", height=140)

if st.button("Comparar", type="primary"):
    textos_a = [l.strip() for l in texto_a.split("\n") if l.strip()]
    textos_b = [l.strip() for l in texto_b.split("\n") if l.strip()]

    if not textos_a or not textos_b:
        st.warning("Preencha os dois grupos para comparar.")
    else:
        df_a = pipeline(carregar_textos(textos_a))
        df_b = pipeline(carregar_textos(textos_b))

        sent_a = df_a["sentimento"].mean()
        sent_b = df_b["sentimento"].mean()

        neg_a = (df_a["sentimento"] < -0.1).mean() * 100
        neg_b = (df_b["sentimento"] < -0.1).mean() * 100

        pos_a = (df_a["sentimento"] > 0.1).mean() * 100
        pos_b = (df_b["sentimento"] > 0.1).mean() * 100

        df_chart = pd.DataFrame({
            "Sentimento Médio": [sent_a, sent_b],
            "% Negativos": [neg_a, neg_b],
            "% Positivos": [pos_a, pos_b],
            "Qtd. Textos": [len(df_a), len(df_b)]
        }, index=["Grupo A", "Grupo B"])

        st.session_state["df_chart"] = df_chart
        st.session_state["df_a"] = df_a
        st.session_state["df_b"] = df_b

        st.dataframe(df_chart.round(3))

        # ======================
        # GRÁFICO DE BARRAS
        # ======================
        st.subheader("📊 Métricas Comparativas")
        st.bar_chart(df_chart[["Sentimento Médio", "% Negativos", "% Positivos"]])

        # ======================
        # HISTOGRAMA
        # ======================
        st.subheader("📉 Histograma de Sentimentos")
        fig_h, ax_h = plt.subplots(figsize=(8, 4))
        ax_h.hist(df_a["sentimento"], bins=15, alpha=0.6, label="Grupo A")
        ax_h.hist(df_b["sentimento"], bins=15, alpha=0.6, label="Grupo B")
        ax_h.axvline(0, color="black", linestyle="--", linewidth=1)
        ax_h.set_title("Distribuição de Sentimentos")
        ax_h.set_xlabel("Score")
        ax_h.set_ylabel("Frequência")
        ax_h.legend()
        st.pyplot(fig_h)

        # ======================
        # CANDLESTICK
        # ======================
        st.subheader("📈 Candlestick de Sentimento")
        fig_c, ax_c = plt.subplots(figsize=(7, 4))
        groups = ["Grupo A", "Grupo B"]
        means = [sent_a, sent_b]

        for i, mean in enumerate(means):
            low = -1
            high = 1
            ax_c.plot([i, i], [low, high], color="black", linewidth=2)
            rect_height = abs(mean)
            rect_bottom = min(0, mean)
            ax_c.add_patch(
                plt.Rectangle((i - 0.2, rect_bottom), 0.4, rect_height,
                              facecolor="tab:blue", edgecolor="black")
            )
            ax_c.text(i, mean + 0.05, f"{mean:.3f}", ha="center", fontweight="bold")

        ax_c.set_xticks([0, 1])
        ax_c.set_xticklabels(groups)
        ax_c.axhline(0, color="gray", linestyle="--")
        ax_c.set_ylabel("Score")
        ax_c.set_title("Amplitude e Tendência do Sentimento")
        st.pyplot(fig_c)

# ======================================================
# PDF — RELATÓRIO ROBUSTO
# ======================================================
def gerar_relatorio_pdf(df_chart):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()

    elements = []
    elements.append(Paragraph("Relatório Analítico de Sentimento — LexInsight", styles["Title"]))
    elements.append(Spacer(1, 20))

    elements.append(Paragraph("1. Visão Geral", styles["Heading2"]))
    elements.append(Paragraph(
        "Este relatório apresenta uma análise comparativa do tom emocional dos textos, "
        "com base em métricas quantitativas e inferência semântica.", styles["Normal"]
    ))

    elements.append(Spacer(1, 20))
    elements.append(Paragraph("2. Métricas Quantitativas", styles["Heading2"]))

    data = [["Grupo"] + list(df_chart.columns)]
    for idx, row in df_chart.iterrows():
        data.append([idx] + [f"{v:.3f}" for v in row])

    table = Table(data)
    table.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 1, colors.black),
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("ALIGN", (1,1), (-1,-1), "CENTER"),
    ]))
    elements.append(table)

    elements.append(Spacer(1, 20))
    elements.append(Paragraph("3. Inferência Analítica", styles["Heading2"]))

    inferencia = (
        "Valores médios negativos indicam linguagem marcada por melancolia, frustração "
        "ou ausência de perspectiva positiva. Distribuições mais concentradas abaixo de zero "
        "sugerem discursos emocionalmente densos e consistentes."
    )
    elements.append(Paragraph(inferencia, styles["Normal"]))

    elements.append(Spacer(1, 20))
    elements.append(Paragraph("4. Conclusão Técnica", styles["Heading2"]))
    elements.append(Paragraph(
        "Os dados demonstram que o sentimento textual não é aleatório, apresentando padrões "
        "identificáveis que podem subsidiar análises estratégicas e comparativas.",
        styles["Normal"]
    ))

    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()

if "df_chart" in st.session_state:
    if st.button("📄 Gerar Relatório PDF"):
        pdf = gerar_relatorio_pdf(st.session_state["df_chart"])
        st.download_button(
            "⬇️ Baixar Relatório PDF",
            data=pdf,
            file_name="Relatorio_LexInsight.pdf",
            mime="application/pdf"
        )

# ======================================================
# ANÁLISE INDIVIDUAL
# ======================================================
st.subheader("🔍 Análise Individual")

texto_usuario = st.text_area("Cole aqui o texto para análise", height=200)

if st.button("Analisar Texto"):
    textos = [l.strip() for l in texto_usuario.split("\n") if l.strip()]

    if textos:
        df = pipeline(carregar_textos(textos))
        st.metric("Sentimento Médio", round(df["sentimento"].mean(), 3))

        tokens = sum(df["tokens"], [])
        wc = WordCloud(width=800, height=400, background_color="white")
        wc.generate(" ".join(tokens))

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc)
        ax.axis("off")
        st.pyplot(fig)
    else:
        st.warning("Insira algum texto válido.")
