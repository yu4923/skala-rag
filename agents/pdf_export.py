"""승인된 보고서 또는 최종 보류 요약을 분량 검증 후 PDF로 내보낸다."""
from io import BytesIO
from pathlib import Path
from html import escape
import os
import re
import pymupdf as fitz
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak

OUTPUTS = Path(__file__).resolve().parents[1] / 'outputs'
WIDTH, HEIGHT = A4
MARGIN = 48
CONTENT_HEIGHT = HEIGHT - 2 * MARGIN


def report_font():
    name = 'ReportKorean'
    if name in pdfmetrics.getRegisteredFontNames():
        return name
    candidates = [os.environ.get('REPORT_FONT_PATH', ''),
                  '/System/Library/Fonts/Supplemental/Arial Unicode.ttf',
                  '/usr/share/fonts/truetype/nanum/NanumGothic.ttf']
    for filename in candidates:
        if filename and Path(filename).is_file():
            pdfmetrics.registerFont(TTFont(name, filename))
            return name
    name = 'HYSMyeongJo-Medium'
    if name not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(UnicodeCIDFont(name))
    return name


def flowables(text, size):
    font = report_font()
    base = ParagraphStyle('body', fontName=font, fontSize=size, leading=size * 1.65,
                          wordWrap='CJK', spaceAfter=8, textColor=colors.HexColor('#243247'))
    result = []
    for block in re.split(r'\n\s*\n', text.strip()):
        for line in block.splitlines():
            if line.strip() == '---':
                continue
            match = re.match(r'^(#{1,3})\s+(.+)', line)
            if match:
                level = len(match[1])
                style = ParagraphStyle('heading', parent=base, fontSize={1:20, 2:14, 3:11}[level],
                                       leading={1:28, 2:22, 3:18}[level], spaceBefore=8,
                                       spaceAfter=10, keepWithNext=True)
                result.append(Paragraph(escape(match[2]), style))
            else:
                value = escape(line)
                value = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', value)
                result.append(Paragraph(value, base))
        result.append(Spacer(1, 4))
    return result


def build_pdf(story, title):
    buffer = BytesIO()
    def footer(canvas, doc):
        canvas.setFont(report_font(), 8)
        canvas.setFillColor(colors.HexColor('#6B7280'))
        canvas.drawString(MARGIN, 25, title)
        canvas.drawRightString(WIDTH-MARGIN, 25, str(doc.page))
    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN,
                            topMargin=MARGIN, bottomMargin=MARGIN, title=title, author='')
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return buffer.getvalue()


def page_count(data):
    with fitz.open(stream=data, filetype='pdf') as doc:
        return len(doc)


def has_orphan_url_page(data):
    with fitz.open(stream=data, filetype='pdf') as doc:
        for page in doc:
            body = '\n'.join(block[4] for block in page.get_text('blocks') if block[3] < HEIGHT - 45).strip()
            if len(body) < 300 and body.startswith(('http://', 'https://')):
                return True
    return False


def save_report_pdf(markdown, company, report_date, *, deferred=False, output_dir=None):
    """본문을 자르지 않고 SUMMARY 높이/각 장/전체 페이지 상한을 검증한다."""
    title = f'{company} 기업 분석 보고서'
    for size in (10.5, 10, 9.5):
        if deferred:
            parts = markdown.split('\n\n---\n\n')
            # 첫 기업 페이지에 제목/개요를 합쳐 회사당 한 페이지를 유지한다.
            sections = [parts[0] + '\n\n' + parts[1]] + parts[2:] if len(parts)>1 else parts
            if any(page_count(build_pdf(flowables(s, size), title)) > 1 for s in sections):
                continue
        else:
            parts = re.split(r'(?=^## )', markdown, flags=re.M)
            front, sections = parts[0], parts[1:]
            summary = next((s for s in sections if s.startswith('## SUMMARY')), '')
            if not summary:
                raise ValueError('PDF SUMMARY가 없습니다.')
            measured = flowables(summary, size)
            height = sum(item.wrap(WIDTH-2*MARGIN-12, CONTENT_HEIGHT)[1] + item.getSpaceBefore() + item.getSpaceAfter() for item in measured)
            if height > CONTENT_HEIGHT / 2:
                continue
            if any(page_count(build_pdf(flowables(s, size), title)) > 2 for s in sections):
                continue
            sections[0] = front + sections[0]
        story = []
        for index, section in enumerate(sections):
            if index:
                story.append(PageBreak())
            story.extend(flowables(section, size))
        data = build_pdf(story, title)
        if page_count(data) <= 10 and not has_orphan_url_page(data):
            directory = Path(output_dir) if output_dir else OUTPUTS
            directory.mkdir(parents=True, exist_ok=True)
            safe_name = re.sub(r'[^\w가-힣.-]+', '_', company).strip('._') or 'report'
            path = directory / f'{safe_name}_{report_date}.pdf'
            temporary = path.with_suffix('.pdf.tmp')
            temporary.write_bytes(data)
            temporary.replace(path)
            return path.resolve()
    raise ValueError('PDF 분량 제한 초과: SUMMARY 반 페이지, 각 장 2페이지, 전체 10페이지 이내로 본문을 줄여야 합니다.')
