#!/usr/bin/env python3
"""Render the editable Turkish Markdown report as a readable, linked A4 PDF.

Dependency: reportlab. Install DejaVu fonts or pass --font-dir.
No GPU packages required. Tables/paragraphs are reflowed, not screenshots.
"""
import argparse
from html import escape
import re
from pathlib import Path
import textwrap
import tempfile

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, LongTable,
                               TableStyle, Preformatted, PageBreak, KeepTogether)

ROOT = Path(__file__).resolve().parents[1]


def font_directory(explicit=None):
    candidates = [Path(explicit)] if explicit else [
        Path('/usr/share/fonts/truetype/dejavu'),
        Path('/usr/local/share/fonts/dejavu'),
        Path.home() / '.cache/codex-runtimes/codex-primary-runtime/dependencies/native/libreoffice-headless/libreoffice/LibreOfficeDev.app/Contents/Resources/fonts/truetype',
    ]
    for folder in candidates:
        if all((folder / n).exists() for n in ['DejaVuSans.ttf', 'DejaVuSans-Bold.ttf', 'DejaVuSansMono.ttf']):
            return folder
    raise SystemExit('DejaVu fonts missing. Install fonts-dejavu-core or pass --font-dir DIR.')


def inline(text):
    text = text.replace('\u2011', '-').replace('\u2013', '-').replace('\u2014', '-')
    text = escape(text)
    text = text.replace('\u2028', '<br/>')
    text = re.sub(r'\[([^\]]+)\]\(([^\s)]+)\)', r'<link href="\2" color="#175a8b">\1</link>', text)
    text = re.sub(r'`([^`]+)`', r'<font name="Mono" size="8.7">\1</font>', text)
    text = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', text)
    return text


class ResearchDoc(SimpleDocTemplate):
    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph) and flowable.style.name == 'Section':
            key = f'section-{getattr(self, "section_count", 0)}'
            self.section_count = getattr(self, 'section_count', 0) + 1
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(flowable.getPlainText(), key, 0, False)


def build(source, destination, fonts, landscape_mode=False, page_offset=0, report_date='13 Kasım 2026'):
    for name, filename in [('Body', 'DejaVuSans.ttf'), ('BodyBold', 'DejaVuSans-Bold.ttf'), ('Mono', 'DejaVuSansMono.ttf')]:
        pdfmetrics.registerFont(TTFont(name, str(fonts / filename)))
    pdfmetrics.registerFontFamily('Body', normal='Body', bold='BodyBold', italic='Body', boldItalic='BodyBold')
    page_size = landscape(A4) if landscape_mode else A4
    width = page_size[0] - 88
    styles = {
        'body': ParagraphStyle('BodyText', fontName='Body', fontSize=9.7, leading=14.6,
                               spaceAfter=7, allowWidows=0, allowOrphans=0),
        'title': ParagraphStyle('Title', fontName='BodyBold', fontSize=22, leading=29,
                                textColor=colors.HexColor('#15344d'), spaceAfter=16, keepWithNext=True),
        'h1': ParagraphStyle('Section', fontName='BodyBold', fontSize=15, leading=20,
                             textColor=colors.HexColor('#15344d'), spaceBefore=16, spaceAfter=9, keepWithNext=True),
        'h2': ParagraphStyle('Subsection', fontName='BodyBold', fontSize=11.5, leading=16,
                             textColor=colors.HexColor('#175a8b'), spaceBefore=10, spaceAfter=6, keepWithNext=True),
        'bullet': ParagraphStyle('Bullet', fontName='Body', fontSize=9.5, leading=14,
                                 leftIndent=12, firstLineIndent=-8, spaceAfter=5),
        'code': ParagraphStyle('Code', fontName='Mono', fontSize=7.8, leading=10.5,
                               leftIndent=8, rightIndent=8, spaceBefore=6, spaceAfter=9,
                               backColor=colors.HexColor('#f2f5f7')),
    }
    story, paragraph = [], []
    lines = source.read_text().splitlines()

    def flush():
        if paragraph:
            content = ' '.join(paragraph)
            paragraph_style = (ParagraphStyle('CodeSource', parent=styles['body'], keepWithNext=True)
                               if content.startswith(('Kaynak:', 'Source:')) else styles['body'])
            story.append(Paragraph(inline(content), paragraph_style))
            paragraph.clear()

    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            flush(); index += 1; continue
        if line.startswith('```'):
            flush(); code = []; index += 1
            while index < len(lines) and not lines[index].strip().startswith('```'):
                for wrapped in textwrap.wrap(lines[index], width=99, replace_whitespace=False,
                                             drop_whitespace=False, break_long_words=True) or ['']:
                    code.append(wrapped)
                index += 1
            code_group = [Preformatted('\n'.join(code), styles['code'])]
            if story and isinstance(story[-1], Paragraph) and story[-1].style.name == 'CodeSource':
                code_group.insert(0, story.pop())
            story.append(KeepTogether(code_group))
            index += 1; continue
        if line.startswith('|') and index + 1 < len(lines) and re.match(r'^\s*\|?[\s:|\-]+\|\s*$', lines[index + 1]):
            flush(); rows = []
            while index < len(lines) and lines[index].strip().startswith('|'):
                row = [x.strip() for x in lines[index].strip().strip('|').split('|')]
                if not all(re.fullmatch(r'[:\- ]+', x) for x in row):
                    rows.append(row)
                index += 1
            count = max(len(row) for row in rows)
            style = ParagraphStyle('Cell', fontName='Body', fontSize=8.1 if count > 4 else 8.6,
                                   leading=11.2 if count > 4 else 12.3, spaceAfter=0)
            cells = [[Paragraph(inline(value), style) for value in row + [''] * (count - len(row))] for row in rows]
            if count >= 7:
                widths = [width / count] * count
            else:
                widths = [width * .27] + [width * .73 / (count - 1)] * (count - 1) if count > 1 else [width]
            table = LongTable(cells, colWidths=widths, repeatRows=1, hAlign='LEFT', splitInRow=1)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#dfeaf1')),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f6f8fa')]),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('LEFTPADDING', (0, 0), (-1, -1), 6), ('RIGHTPADDING', (0, 0), (-1, -1), 6),
                ('TOPPADDING', (0, 0), (-1, -1), 6), ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('LINEBELOW', (0, 0), (-1, 0), .6, colors.HexColor('#9db4c4')),
                ('LINEBELOW', (0, 1), (-1, -1), .25, colors.HexColor('#dce3e8')),
            ]))
            story.extend([table, Spacer(1, 9)]); continue
        if line.startswith('#'):
            flush(); level = len(line) - len(line.lstrip('#')); content = line.lstrip('#').strip()
            if content == 'Kaynakça':
                story.append(PageBreak())
            chosen = 'title' if level == 1 and not story else ('h1' if level <= 2 else 'h2')
            story.append(Paragraph(inline(content), styles[chosen])); index += 1; continue
        if line in ['---', '***']:
            flush(); story.append(Spacer(1, 6)); index += 1; continue
        if line.startswith('>'):
            flush(); story.append(Paragraph(inline(line.lstrip('> ').strip()), styles['bullet'])); index += 1; continue
        if re.match(r'^[-*]\s+', line):
            flush(); story.append(Paragraph('- ' + inline(line[2:]), styles['bullet'])); index += 1; continue
        if re.match(r'^\d+[.)]\s+', line):
            flush(); story.append(Paragraph(inline(line), styles['bullet'])); index += 1; continue
        paragraph.append(line + ('\u2028' if lines[index].endswith('  ') else '')); index += 1
    flush()

    def page(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor('#d5dfe6'))
        canvas.line(44, page_size[1] - 35, page_size[0] - 44, page_size[1] - 35)
        canvas.setFont('Body', 7.5); canvas.setFillColor(colors.HexColor('#587080'))
        canvas.drawString(44, page_size[1] - 27, 'YZM5110 | HAMMER + CRAF | Araştırma raporu')
        canvas.drawString(44, 25, report_date + ' | Sunucu kayıtları ve doğrulama sınırları açıkça ayrılmıştır')
        canvas.drawRightString(page_size[0] - 44, 25, str(doc.page + page_offset))
        canvas.restoreState()

    destination.parent.mkdir(parents=True, exist_ok=True)
    doc = ResearchDoc(str(destination), pagesize=page_size, rightMargin=44, leftMargin=44,
                      topMargin=49, bottomMargin=44, title='HAMMER Üzerinde Çok Modlu Karar Düzeltmesi',
                      author='Muhammed Emir Koçak', allowSplitting=1)
    doc.build(story, onFirstPage=page, onLaterPages=page)
    print(destination)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=ROOT / 'reports/Kocak_YZM5110_Research.md')
    p.add_argument('--output', type=Path, default=ROOT / 'reports/Kocak_YZM5110_Research.pdf')
    p.add_argument('--font-dir')
    p.add_argument('--report-date', default='13 Kasım 2026')
    p.add_argument('--without-matrix', action='store_true', help='Render just the main Markdown, without appendix')
    args = p.parse_args()
    fonts = font_directory(args.font_dir)
    matrix_path = ROOT / 'docs/LITERATURE_MATRIX.md'
    if args.without_matrix or args.source.name != 'Kocak_YZM5110_Research.md' or not matrix_path.exists():
        build(args.source, args.output, fonts, report_date=args.report_date)
    else:
        from pypdf import PdfReader, PdfWriter
        rows = [line.replace('Sayım ve bölme için aşağıdaki ortak protokol notu geçerli.',
                             'Sayım/protokol notları rapor §6 ve literatür dosyasındadır.')
                for line in matrix_path.read_text().splitlines() if line.startswith('|')]
        if len(rows) < 7:
            raise ValueError('Expected a header and at least five paper rows in literature matrix')
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            main, appendix = folder / 'main.pdf', folder / 'appendix.pdf'
            appendix_source = folder / 'appendix.md'
            appendix_source.write_text('# Ek A. Eleştirel literatür karşılaştırma matrisi\n\n'
                'Ana çalışma R1 HAMMER; R1-R5 yayımlanmış çalışmalar, R6 isteğe bağlı ön baskıdır. '
                'Kaynak bulgusu ile yorum ayrımı hücrelerde korunmuştur. Sürüm/erişim ayrıntıları ve '
                'öğrencinin okuma kartları paketin docs klasöründedir.\n\n' + '\n'.join(rows) + '\n')
            build(args.source, main, fonts, report_date=args.report_date)
            offset = len(PdfReader(main).pages)
            build(appendix_source, appendix, fonts, landscape_mode=True, page_offset=offset, report_date=args.report_date)
            writer = PdfWriter()
            writer.append(main); writer.append(appendix)
            writer.add_metadata({'/Title': 'Dondurulmuş HAMMER Üzerinde Çok Modlu Karar Düzeltmesi', '/Author': 'Muhammed Emir Koçak'})
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open('wb') as f:
                writer.write(f)
            print(args.output)
