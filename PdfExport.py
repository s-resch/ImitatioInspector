from fpdf import FPDF
from fpdf.fonts import FontFace
from fpdf.enums import TableCellFillMode
from datetime import datetime
import logging


class PdfExport(FPDF):
    def __init__(self) -> None:
        super().__init__()

        logger = logging.getLogger("fpdf")
        logger.setLevel(logging.ERROR)

        self.add_page(format="A4")
        self.add_font("dejavu-serif", style="", fname=r"Fonts\DejaVuSerif.ttf")
        self.add_font("dejavu-serif", style="b",
                      fname=r"Fonts\DejaVuSerif-Bold.ttf")
        self.add_font("dejavu-serif", style="i",
                      fname=r"Fonts\DejaVuSerif-Italic.ttf")
        self.add_font("dejavu-serif", style="bi",
                      fname=r"Fonts\DejaVuSerif-BoldItalic.ttf")
        # Different type of the same font design.
        self.add_font("dejavu-serif-narrow", style="",
                      fname=r"Fonts\DejaVuSerifCondensed.ttf")
        self.add_font("dejavu-serif-narrow", style="i",
                      fname=r"Fonts\DejaVuSerifCondensed-Italic.ttf")

    def CreateHeader(self, title):
        # Setting font: times new roman bold 14
        self.set_font("dejavu-serif", "BU", 14)
        # Moving cursor to the right:
        self.cell(80)
        # Printing title:
        self.cell(30, 10, title, align="C")
        # Performing a line break:
        self.ln(20)

    def CreateFooter(self):
        # Position cursor at 1.5 cm from bottom:
        self.set_y(275)
        # Setting font: arial italic 8
        self.set_font("dejavu-serif", "I", 8)
        # Printing page number:
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def CreateTitle(self, title, author="not indicated", date=datetime.now().strftime("%d/%m/%Y %H:%M:%S"), thresholdDante=None, thresholdPetrarca=None):
        # Setting font: times new roman 12
        self.set_font("dejavu-serif", "", 12)
        # Setting background color
        self.set_fill_color(200, 220, 255)
        # Printing justified text:
        self.multi_cell(0, 5, f"Title: {title}", align="L")
        # Performing a line break:
        self.ln()
        self.multi_cell(0, 5, f"Author: {author}", align="L")
        # Performing a line break:
        self.ln()
        self.multi_cell(0, 5, f"Date: {date}", align="L")
        self.ln()
        self.multi_cell(0, 5, f"Threshold Dante: {thresholdDante}", align="L")
        self.ln()
        self.multi_cell(
            0, 5, f"Threshold Petrarca: {thresholdPetrarca}", align="L")
        # Performing a line break:
        self.ln()
        self.ln()
        self.ln()

    def CreateSummary(self, summary):
        # Setting font: times new roman 12
        self.set_font("dejavu-serif", "BU", 12)
        # Printing justified text:
        self.multi_cell(0, 5, "Summary:", align="L",)
        # Performing a line break:
        self.ln()
        # Setting font: times new roman 12
        self.set_font("dejavu-serif", "", 12)
        # Printing justified text:
        self.multi_cell(
            0, 5, f"Number of Dantesque verses: {str(summary.numberDante)}", align="L")
        # Performing a line break:
        self.ln()
        self.multi_cell(
            0, 5, f"Number of Petrarchesque verses: {str(summary.numberPetrarca)}", align="L")
        # Performing a line break:
        self.ln()
        self.multi_cell(
            0, 5, f"Percentage of Dantesque verses (approx.): {str(round(summary.relativeDante*100, 3))}", align="L")
        # Performing a line break:
        self.ln()
        self.multi_cell(
            0, 5, f"Percentage of Petrarchesque verses (approx.): {str(round(summary.relativePetrarca*100, 3))}", align="L")
        # Performing a line break:
        self.ln()
        self.ln()
        self.ln()

    def CreateTable(self, data):
        # Setting font: times new roman 12
        self.set_font("dejavu-serif", "BU", 12)
        # Printing justified text:
        self.multi_cell(0, 5, "Details:", align="L")
        # Performing a line break:
        self.ln()
        # Setting font: times new roman 12
        self.set_font("dejavu-serif", "", 12)
        # danteStyle = (255, 91, 97)
        danteStyle = (255, 157, 160)
        # petrarchStyle = (193, 229, 252)
        petrarchStyle = (195, 251, 179)
        neutralStyle = (230, 230, 230)
        self.set_draw_color(0, 0, 0)
        self.set_line_width(0.3)
        headingsStyle = FontFace(
            emphasis="BOLD", color=255, fill_color=(100, 100, 100))
        with self.table(
            borders_layout="HORIZONTAL_LINES",
            cell_fill_mode=TableCellFillMode.ROWS,
            headings_style=headingsStyle,
            col_widths=(20, 80, 35, 45),
            line_height=6,
            text_align=("LEFT", "LEFT", "CENTER", "CENTER"),
            width=180,
        ) as table:
            headerRow = table.row()
            headerRow.cell("#")
            headerRow.cell("Verse")
            headerRow.cell("Label")
            headerRow.cell("Probability")
            counter = 1
            for data_row in data:
                row = table.row()
                if data_row.label == "Dante":
                    self.set_fill_color(danteStyle)
                elif data_row.label == "Petrarca":
                    self.set_fill_color(petrarchStyle)
                else:
                    self.set_fill_color(neutralStyle)

                row.cell(str(counter))
                row.cell(data_row.verse)
                row.cell(data_row.label)
                row.cell(str(round(data_row.probability, 10))
                         if data_row.probability > 0 else "")

                counter = counter + 1

    def createResultPdf(self, title, data, author, summary, thresholdDante, thresholdPetrarca, linearcorr=False):
        self.CreateHeader(title)
        self.CreateTitle(title, author, thresholdDante=thresholdDante,
                         thresholdPetrarca=thresholdPetrarca)
        self.CreateSummary(summary)
        self.CreateTable(data)
        # self.CreateFooter()
        if linearcorr:
            filename = "Results_" + title + "_lc_" + \
                str(thresholdDante).replace('.', '') + "_" + \
                str(thresholdPetrarca).replace('.', '') + ".pdf"
        else:
            filename = "Results_" + title +  \
                str(thresholdDante).replace('.', '') + "_" + \
                str(thresholdPetrarca).replace('.', '') + ".pdf"
        self.output(filename)


class ResultSummary:
    def __init__(self, numberDante, numberPetrarca, relativeDante, relativePetrarca) -> None:
        self.numberDante = numberDante
        self.numberPetrarca = numberPetrarca
        self.relativeDante = relativeDante
        self.relativePetrarca = relativePetrarca
