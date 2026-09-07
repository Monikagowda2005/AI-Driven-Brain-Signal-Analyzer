from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch
from datetime import datetime

def generate_report(
    filename,
    prediction,
    confidence,
    channels,
    sampling_rate,
    duration,
    probabilities
):

    doc = SimpleDocTemplate(filename)

    styles = getSampleStyleSheet()

    story = []

    title = Paragraph(
        "<b><font size=18>AI Driven Brain Signal Analyzer Report</font></b>",
        styles["Title"]
    )

    story.append(title)
    story.append(Spacer(1, 0.3 * inch))

    story.append(
        Paragraph(
            f"<b>Date:</b> {datetime.now()}",
            styles["Normal"]
        )
    )

    story.append(Spacer(1, 0.2 * inch))

    table_data = [

        ["Prediction", prediction],

        ["Confidence", f"{confidence:.2f}%"],

        ["Channels", str(channels)],

        ["Sampling Frequency", f"{sampling_rate} Hz"],

        ["Recording Length", f"{duration:.2f} sec"]

    ]

    table = Table(table_data, colWidths=[200, 220])

    table.setStyle(TableStyle([

        ("BACKGROUND", (0, 0), (-1, 0), colors.lightblue),

        ("GRID", (0, 0), (-1, -1), 1, colors.black),

        ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),

        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),

        ("BOTTOMPADDING", (0, 0), (-1, -1), 8)

    ]))

    story.append(table)

    story.append(Spacer(1, 0.4 * inch))

    story.append(
        Paragraph(
            "<b>Class Probabilities</b>",
            styles["Heading2"]
        )
    )

    probability_names = [
        "Rest",
        "Left Fist",
        "Right Fist",
        "Both Fists",
        "Both Feet"
    ]

    prob_table = []

    for name, value in zip(probability_names, probabilities):

        prob_table.append(
            [name, f"{value*100:.2f}%"]
        )

    t = Table(prob_table, colWidths=[220, 200])

    t.setStyle(TableStyle([

        ("GRID", (0, 0), (-1, -1), 1, colors.grey),

        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey)

    ]))

    story.append(t)

    story.append(Spacer(1, 0.3 * inch))

    story.append(
        Paragraph(
            "Generated using AI Driven Brain Signal Analyzer based on Deep Neural Networks.",
            styles["Italic"]
        )
    )

    doc.build(story)

    return filename