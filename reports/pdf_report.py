from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from datetime import datetime

def gerar_relatorio_pdf(df, caminho="relatorio_lexinsight.pdf"):
    c = canvas.Canvas(caminho, pagesize=A4)
    width, height = A4

    sentimento_medio = round(df["sentimento"].mean(), 3)
    total = len(df)

    positivos = (df["sentimento"] > 0.1).sum()
    negativos = (df["sentimento"] < -0.1).sum()
    neutros = total - positivos - negativos

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "Relatório LexInsight")

    c.setFont("Helvetica", 10)
    c.drawString(50, height - 80, f"Data: {datetime.now().strftime('%d/%m/%Y')}")

    c.setFont("Helvetica", 12)
    c.drawString(50, height - 120, f"Sentimento médio: {sentimento_medio}")

    c.drawString(50, height - 150, f"Positivos: {positivos}")
    c.drawString(50, height - 170, f"Neutros: {neutros}")
    c.drawString(50, height - 190, f"Negativos: {negativos}")

    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, height - 240, "Conclusão Automática:")

    c.setFont("Helvetica", 11)
    if sentimento_medio < -0.1:
        conclusao = "O conjunto textual apresenta tendência predominantemente negativa."
    elif sentimento_medio > 0.1:
        conclusao = "O conjunto textual apresenta tendência predominantemente positiva."
    else:
        conclusao = "O conjunto textual apresenta tom majoritariamente neutro."

    c.drawString(50, height - 270, conclusao)

    c.save()