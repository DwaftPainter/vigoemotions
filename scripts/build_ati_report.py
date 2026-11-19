"""Populate the instructor's Word template from verified, text-free project evidence."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt
from lxml import etree
from pypdf import PdfReader

from build_final_paper import (ROOT, OUT, checksum, evidence, fill_template,
                               manuscript_values, markdown_table, read_json)
from common import scores

TEMPLATE = ROOT/'docs/ATI_Final_Report_Template.docx'
CONTENT = ROOT/'docs/ati_report_content.md'
BASELINE = ROOT/'reports/kaggle_equal20_2026-11-05'


def dataset_split_table():
    config = read_json(ROOT/'configs/final_thresholds.json')
    profile = pd.read_csv(OUT/'tables/threshold_profile.csv').set_index('label')
    rows = []
    for split, name in (('train', 'Train'), ('val', 'Validation'), ('test', 'Test / held-out')):
        samples = config['data']['splits'][split]['n']
        if split == 'train':
            positives = int(profile['train_count'].sum())
            neutral = int(profile.loc['neutral', 'train_count'])
        else:
            comparison = pd.read_csv(OUT/f'tables/{split}_comparison.csv')
            neutral_scores = pd.read_csv(OUT/f'tables/{split}_neutral.csv')
            positives = int(comparison.query("strategy == 'fixed' and scope == 'all28'")['positive_support'].item())
            neutral = int(neutral_scores.query("strategy == 'fixed'")['support'].item())
        rows.append({
            'Split / domain': name,
            'Samples': f'{samples:,}',
            'Classes or labels': config['model']['label_count'],
            'Notes': (f'{positives:,} positive assignments, including {neutral:,} Neutral positives. '
                      f'Mean cardinality is {positives / samples:.3f}.'),
        })
    return markdown_table(pd.DataFrame(rows))


def report_values():
    meta = read_json(ROOT/'docs/final_paper_metadata.json')
    values = manuscript_values(meta)
    values['data_summary'] = dataset_split_table()
    baseline = pd.read_csv(BASELINE/'model_comparison.csv')
    names = {'tfidf_lr':'TF-IDF + LR', 'gru':'BiGRU', 'phobert':'PhoBERT'}
    rows, training, budgets = [], [], []
    for row in baseline.itertuples():
        rows.append({'Model':names[row.model], 'Outputs':row.label_count, 'Macro-F1':row.macro_f1,
                     'Micro-F1':row.micro_f1, 'Hamming':row.hamming_loss, 'LRAP':row.lrap})
        model = f'{names[row.model]}-{row.label_count}'
        run = read_json(BASELINE/f'evidence/{row.label_count}/{row.model}/run.json')
        settings = run['config'].get(row.model, {})
        training.append({'Model':model, 'Checkpoint':settings.get('checkpoint', 'From scratch'),
                         'Input size':f"{settings['max_length']} tokens" if settings else '≤40k features',
                         'Epochs':settings.get('epochs', 'N/A'), 'Batch':settings.get('batch_size', 'N/A'),
                         'LR':f"{settings['learning_rate']:.0e}" if settings else 'N/A',
                         'Params':f'{row.parameter_count:,}'})
        budgets.append({'Model':model,
                        'Selected epoch':str(int(row.best_epoch)) if pd.notna(row.best_epoch) else 'N/A',
                        'Params':f'{row.parameter_count:,}',
                        'Training time (s)':f'{row.train_seconds:.3f}'})
    values['baseline_results'] = markdown_table(pd.DataFrame(rows))
    values['training_configuration'] = markdown_table(pd.DataFrame(training))
    values['baseline_training'] = markdown_table(pd.DataFrame(budgets))
    values['rq_table'] = markdown_table(pd.DataFrame([{
        'RQ':'RQ1', 'Question':'Does validation-selected threshold tuning improve held-out all28 Macro-F1?',
        'Hypothesis':'Global/per-label tuning improves test F1 over fixed 0.5.',
        'Control (Midterm)':'Midterm PhoBERT-28 checkpoint, fixed threshold 0.5 for all outputs.'}]))
    all28 = pd.read_csv(OUT/'tables/test_comparison.csv').query("scope == 'all28'").set_index('strategy')
    fixed = all28.loc['fixed', 'macro_f1']
    best = all28['macro_f1'].max()
    comparison = []
    for strategy, decoder in (('global', 'global'), ('per_label', 'per-label')):
        tuned = all28.loc[strategy, 'macro_f1']
        comparison.append({'Model':f'PhoBERT-28 ({decoder})',
                           'Fixed 0.5 (A)':f'{fixed:.6f}',
                           'Tuned cutoff (B)':f'**{tuned:.6f}**' if tuned == best else f'{tuned:.6f}',
                           'Δ (B − A)':f'{100 * (tuned - fixed):+.4f} pp'})
    values['rq1_template_comparison'] = markdown_table(pd.DataFrame(comparison))
    cases = pd.read_csv(OUT/'human_audit/cases.csv',dtype=str,keep_default_na=False).set_index('sample_id')
    values['failure_examples'] = markdown_table(pd.DataFrame([
        {'Case ID':key, 'Reference labels':cases.loc[key,'true_labels'].replace(';',', '),
         'Per-label prediction':cases.loc[key,'pred_per_label_labels'].replace(';',', ')}
        for key in ('you003220','tik017852','you001985','247','18169')]))
    authors = meta.get('authors') or [{} for _ in range(3)]
    values['member_contributions'] = markdown_table(pd.DataFrame([
        {'Student ID':a.get('student_id') or '[Student ID]', 'Full name':a.get('name') or '[Full name]',
         'Main tasks':a.get('tasks') or '[Main tasks]', '%':a.get('percentage','[%]')}
        for a in authors]))
    values['ai_declaration'] = markdown_table(pd.DataFrame([
        {'Tool':'[Tool and version]', 'Purpose':'[Purpose]', 'Sections or files affected':'[Sections or files]'}]))
    return values


def verify_baselines():
    """Recalculate the six reported baseline rows without model loading."""
    inputs = {str(BASELINE/'model_comparison.csv'):checksum(BASELINE/'model_comparison.csv')}
    for row in pd.read_csv(BASELINE/'model_comparison.csv').itertuples():
        folder = BASELINE/f'evidence/{row.label_count}/{row.model}'
        path = folder/'val_predictions.npz'
        with np.load(path,allow_pickle=False) as packet:
            actual = scores(packet['true_label_vector'],packet['probability_vector'])
        for key in ('macro_f1','micro_f1','hamming_loss','lrap'):
            if not np.isclose(actual[key],getattr(row,key),rtol=0,atol=1e-12):
                raise ValueError(f'Baseline metric mismatch: {row.label_count}/{row.model}/{key}')
        for file in (path,folder/'run.json'):
            inputs[str(file)] = checksum(file)
    return inputs


def figures(destination):
    assets = destination/'ati_report_assets'; assets.mkdir(exist_ok=True)
    profile = pd.read_csv(OUT/'tables/threshold_profile.csv')
    with plt.rc_context({'font.size':9, 'axes.spines.top':False, 'axes.spines.right':False}):
        data = profile[profile.label!='neutral'].sort_values('train_count',ascending=False,kind='stable')
        fig,ax = plt.subplots(figsize=(7,5.7))
        ax.barh(data.label,data.train_count,color='.75',edgecolor='black',linewidth=.5)
        ax.invert_yaxis(); ax.set_xlabel('Positive retained-training examples')
        ax.set_ylabel('Emotion label'); ax.tick_params(axis='y',labelsize=8)
        fig.tight_layout()
        for ext in ('png','pdf'):fig.savefig(assets/f'train_label_frequency.{ext}',dpi=220)
        plt.close(fig)
        val = pd.read_csv(OUT/'tables/val_comparison.csv').query("scope=='all28'").set_index('strategy')
        test = pd.read_csv(OUT/'tables/test_comparison.csv').query("scope=='all28'").set_index('strategy')
        methods = ['fixed','global','per_label']; x=np.arange(3)
        fig,ax=plt.subplots(figsize=(7,3.5))
        ax.plot(x,[val.loc[s,'macro_f1'] for s in methods],'o-',color='black',label='Validation')
        ax.plot(x,[test.loc[s,'macro_f1'] for s in methods],'s--',color='.45',label='Test')
        for i,s in enumerate(methods):
            ax.annotate(f"{test.loc[s,'macro_f1']:.6f}",(i,test.loc[s,'macro_f1']),xytext=(0,7),textcoords='offset points',ha='center')
        ax.set_xticks(x,['Fixed 0.5','Global 0.4','Per-label']);ax.set_ylabel('Macro-F1 (all28)')
        ax.set_ylim(.565,.607);ax.grid(axis='y',alpha=.25);ax.legend(frameon=False)
        fig.tight_layout()
        for ext in ('png','pdf'):fig.savefig(assets/f'validation_test_bw.{ext}',dpi=220)
        plt.close(fig)
        fig,ax=plt.subplots(figsize=(7,5.8))
        ax.scatter(profile.threshold,profile.label,color='black',s=19,zorder=3)
        ax.axvline(.5,color='.25',linestyle=':',label='Fixed 0.5')
        ax.axvline(.4,color='.5',linestyle='--',label='Global 0.4')
        ax.invert_yaxis();ax.set_xlim(0,.85);ax.set_xlabel('Validation-selected cutoff')
        ax.tick_params(axis='y',labelsize=8);ax.grid(axis='x',alpha=.2);ax.legend(loc='lower right',frameon=False)
        fig.tight_layout()
        for ext in ('png','pdf'):fig.savefig(assets/f'threshold_profile_bw.{ext}',dpi=220)
        plt.close(fig)
    return sorted(assets.iterdir())


def field(paragraph,instruction,result):
    begin = OxmlElement('w:fldChar');begin.set(qn('w:fldCharType'),'begin');begin.set(qn('w:dirty'),'true')
    paragraph.add_run()._r.append(begin)
    code=OxmlElement('w:instrText');code.set(qn('xml:space'),'preserve');code.text=' '+instruction+' '
    paragraph.add_run()._r.append(code)
    separate=OxmlElement('w:fldChar');separate.set(qn('w:fldCharType'),'separate');paragraph.add_run()._r.append(separate)
    paragraph.add_run(str(result))
    end=OxmlElement('w:fldChar');end.set(qn('w:fldCharType'),'end');paragraph.add_run()._r.append(end)


def add_inline(paragraph,text):
    for token in re.split(r'(\*\*.+?\*\*|`[^`]+`)',text):
        run=paragraph.add_run(token[2:-2] if token.startswith('**') else token[1:-1] if token.startswith('`') else token)
        if token.startswith('**'):run.bold=True


def equation(document,key,number,style):
    p=document.add_paragraph(style=style)
    p.alignment=WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.tab_stops.add_tab_stop(Cm(8),WD_TAB_ALIGNMENT.CENTER)
    p.paragraph_format.tab_stops.add_tab_stop(Cm(15.9),WD_TAB_ALIGNMENT.RIGHT)
    def run(text):
        r=OxmlElement('m:r');t=OxmlElement('m:t');t.text=text;r.append(t);return r
    def sub(base,index):
        item=OxmlElement('m:sSub')
        for tag,text in [('m:e',base),('m:sub',index)]:
            node=OxmlElement(tag);node.append(run(text));item.append(node)
        return item
    def fraction(numerator,denominator):
        item=OxmlElement('m:f')
        for tag,children in [('m:num',numerator),('m:den',denominator)]:
            node=OxmlElement(tag);node.extend(children);item.append(node)
        return item
    def summation(index,upper,children):
        item=OxmlElement('m:nary');properties=OxmlElement('m:naryPr')
        symbol=OxmlElement('m:chr');symbol.set(qn('m:val'),'∑');properties.append(symbol)
        if not upper:
            hide=OxmlElement('m:supHide');hide.set(qn('m:val'),'1');properties.append(hide)
        item.append(properties)
        for tag,text in [('m:sub',index),('m:sup',upper)]:
            node=OxmlElement(tag)
            if text:node.append(run(text))
            item.append(node)
        expression=OxmlElement('m:e');expression.extend(children);item.append(expression);return item
    def total(name):return summation('c=1','C',[sub(name,'c')])
    formulas={
        'sigmoid':lambda:[sub('p','i,c'),run(' = σ('),sub('z','i,c'),run('),  '),sub('ŷ','i,c'),run(' = 1['),sub('p','i,c'),run(' ≥ '),sub('τ','c'),run(']')],
        'bce':lambda:[run('ℒ = −'),fraction([run('1')],[run('NC')]),summation('i=1','N',[
            summation('c=1','C',[run('['),sub('y','i,c'),run(' ln '),sub('p','i,c'),run(' + (1 − '),sub('y','i,c'),run(') ln(1 − '),sub('p','i,c'),run(')]')])])],
        'macro':lambda:[sub('F1','c'),run(' = '),fraction([run('2'),sub('TP','c')],[run('2'),sub('TP','c'),run(' + '),sub('FP','c'),run(' + '),sub('FN','c')]),run('; Macro-F1 = '),fraction([total('F1')],[run('C')])],
        'micro':lambda:[run('Micro-F1 = '),fraction([run('2'),total('TP')],[run('2'),total('TP'),run(' + '),total('FP'),run(' + '),total('FN')])],
        'hamming':lambda:[run('Hamming Loss = '),fraction([run('1')],[run('NC')]),summation('i=1','N',[summation('c=1','C',[run('1['),sub('ŷ','i,c'),run(' ≠ '),sub('y','i,c'),run(']')])])],
        'lrap':lambda:[run('LRAP = '),fraction([run('1')],[run('N')]),summation('i=1','N',[
            fraction([run('1')],[run('|'),sub('Y','i'),run('|')]),summation('c∈Yᵢ','',[fraction([sub('r⁺','i,c')],[sub('r','i,c')])])])],
        'precision':lambda:[run('Precision = '),fraction([run('TP')],[run('TP + FP')]),run(',   Recall = '),fraction([run('TP')],[run('TP + FN')])],
        'thresholds':lambda:[run('τ* = '),sub('arg max','t∈G'),run(' Macro-F1(val; t);  '),sub('τ*','c'),run(' = '),sub('arg max','t∈G'),sub(' F1','c'),run('(val; t)')],
    }
    p.add_run('\t');math=OxmlElement('m:oMath');math.extend(formulas[key]());p._p.append(math)
    p.add_run('\t(');field(p,'SEQ Equation \\* ARABIC',number);p.add_run(')')


def render_word(markdown,destination,asset_base):
    document=Document(TEMPLATE);body=document.element.body
    sample_tables = {tuple(c.text for c in t.rows[0].cells):
                     (deepcopy(t._tbl.tblPr), [c.width for c in t.columns]) for t in document.tables}
    contribution_style = deepcopy(next(p._p.pPr for p in document.paragraphs if p.text=='[Contribution 1]'))
    elements=list(body)
    breaks=[deepcopy(e) for e in elements if e.find('.//'+qn('w:sectPr')) is not None]
    toc=deepcopy(next(e for e in elements if e.tag==qn('w:sdt')))
    first_break=next(i for i,e in enumerate(elements) if e.find('.//'+qn('w:sectPr')) is not None)
    for element in elements[first_break+1:-1]:body.remove(element)
    styles={s.style_id:s for s in document.styles}
    for sid,level in [('Heading1',0),('Heading2',1),('Heading3',2)]:
        properties=styles[sid].element.get_or_add_pPr()
        outline=properties.find(qn('w:outlineLvl'))
        if outline is None:outline=OxmlElement('w:outlineLvl');properties.append(outline)
        outline.set(qn('w:val'),str(level))
    title=markdown.splitlines()[0].lstrip('# ')
    replacements={'[PROJECT TITLE]':title,'[P00]':'P02','[Month YYYY]':'November 2026','[Full name]*':'[Full name]'}
    for element in list(body)[:first_break+1]:
        for node in element.iter(qn('w:t')):
            for old,new in replacements.items():
                if node.text:node.text=node.text.replace(old,new)
    document.core_properties.title=title;document.core_properties.author='';document.core_properties.last_modified_by=''
    for marker in ('commentRangeStart','commentRangeEnd','commentReference'):
        for node in body.iter(qn('w:'+marker)):node.getparent().remove(node)
    for rid,relationship in list(document.part.rels.items()):
        if relationship.reltype.endswith('/comments'):document.part.drop_rel(rid)
    lines=markdown.splitlines();i=1;front=True;equation_number=0
    while i<len(lines):
        line=lines[i].strip();i+=1
        if not line:continue
        level=len(line)-len(line.lstrip('#'))
        if level:
            text=line[level:].strip()
            if text.startswith('1  Introduction'):
                document.add_page_break()
                document.add_paragraph('Table of Contents',style=styles['FrontHeading'])
                body.insert(len(body)-1,toc)
                body.insert(len(body)-1,deepcopy(breaks[1]))
                front=False
            p=document.add_paragraph(text,style=styles['FrontHeading' if front else 'Heading1' if level==2 else 'Heading2'])
            if text=='References' or text.startswith('Appendix '):p.paragraph_format.page_break_before=True
            continue
        eq=re.fullmatch(r'<!-- equation:([a-z]+) -->',line)
        if eq:
            equation_number+=1;equation(document,eq[1],equation_number,styles['Normal']);continue
        caption=re.fullmatch(r'\*\*Table (A\.\d+|\d+)\. (.*?)\*\*',line)
        if caption:
            p=document.add_paragraph(style=styles['Caption']);p.paragraph_format.keep_with_next=True
            appendix, _, number = caption[1].rpartition('.')
            p.add_run('Table A.' if appendix else 'Table ')
            field(p,'SEQ TableAppendixA \\* ARABIC' if appendix else 'SEQ Table \\* ARABIC',number)
            p.add_run(': '+caption[2]);continue
        if line.startswith('|'):
            rows=[]
            while True:
                cells=[c.strip() for c in line.strip('|').split('|')]
                if not all(re.fullmatch(r'[-: ]+',cell) for cell in cells):rows.append(cells)
                if i>=len(lines) or not lines[i].strip().startswith('|'):break
                line=lines[i].strip();i+=1
            table=document.add_table(rows=0,cols=len(rows[0]));table.autofit=False
            width=document.sections[-1].page_width-document.sections[-1].left_margin-document.sections[-1].right_margin
            weights=[1.4]+[1]*(len(rows[0])-1) if len(rows[0])>2 else [1,2.3]
            widths=[int(width*w/sum(weights)) for w in weights]
            template_header = tuple(rows[0])
            if template_header == ('Model','Fixed 0.5 (A)','Tuned cutoff (B)','Δ (B − A)'):
                template_header = ('Model','[Condition A]','[Condition B]','Δ (B − A)')
            if template_header in sample_tables:
                properties, widths = sample_tables[template_header]
                table._tbl.replace(table._tbl.tblPr, deepcopy(properties))
                table.autofit=False
                if template_header == ('Model','Checkpoint','Input size','Epochs','Batch','LR','Params'):
                    # Allow full checkpoint identifiers and parameter counts in the template columns.
                    widths = [Cm(value) for value in (2.95, 4, 2.1, 1.9, 1.35, 1.3, 2.4)]
            for column,w in zip(table.columns,widths):column.width=w
            if template_header not in sample_tables:
                borders=OxmlElement('w:tblBorders')
                for edge in ('top','bottom','insideH'):
                    item=OxmlElement('w:'+edge);item.set(qn('w:val'),'single');item.set(qn('w:sz'),'4');item.set(qn('w:color'),'808080');borders.append(item)
                table._tbl.tblPr.append(borders)
            for n,row in enumerate(rows):
                item=table.add_row();props=item._tr.get_or_add_trPr();props.append(OxmlElement('w:cantSplit'))
                if n==0:props.append(OxmlElement('w:tblHeader'))
                for cell,value,w in zip(item.cells,row,widths):
                    cell.width=w;p=cell.paragraphs[0];p.style=styles['TableText'];add_inline(p,value)
                    if len(rows)<=9 and n<len(rows)-1:
                        p.paragraph_format.keep_with_next=True
                    if n==0:
                        for run in p.runs:run.bold=True
            continue
        image=re.fullmatch(r'!\[Figure (A\.\d+|\d+)\. (.*?)\]\((.*?)\)',line)
        if image:
            number,caption,relative=image.groups()
            p=document.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.paragraph_format.keep_with_next=True
            p.add_run().add_picture(str(asset_base/relative),width=Cm(15.7))
            appendix, _, number = number.rpartition('.')
            p=document.add_paragraph(style=styles['Caption']);p.add_run('Figure A.' if appendix else 'Figure ')
            field(p,'SEQ FigureAppendixA \\* ARABIC' if appendix else 'SEQ Figure \\* ARABIC',number)
            p.add_run(': '+caption);continue
        if line.startswith('- '):
            p=document.add_paragraph()
            p._p.insert(0,deepcopy(contribution_style))
            add_inline(p,line[2:]);continue
        if line.startswith('```'):
            block=[]
            while i<len(lines) and not lines[i].startswith('```'):block.append(lines[i]);i+=1
            i+=1;document.add_paragraph('\n'.join(block),style=styles['Code']);continue
        paragraph=[line]
        while i<len(lines) and lines[i].strip() and not lines[i].lstrip().startswith(('#','|','![','**Table','<!--','```')):
            paragraph.append(lines[i].strip());i+=1
        text=' '.join(paragraph)
        style=styles['Reference'] if re.match(r'^\[\d+\]',text) else styles['Normal']
        add_inline(document.add_paragraph(style=style),text)
    update=document.settings.element.find(qn('w:updateFields'))
    if update is None:update=OxmlElement('w:updateFields');document.settings.element.append(update)
    update.set(qn('w:val'),'true')
    document.save(destination)


def build(destination=ROOT/'reports'):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    inputs,audit=evidence();inputs.update(verify_baselines())
    for path in (TEMPLATE,CONTENT,ROOT/'docs/final_paper_metadata.json',Path(__file__).resolve(),ROOT/'scripts/build_final_paper.py'):
        inputs[str(path)]=checksum(path)
    text=fill_template(CONTENT.read_text(),report_values())
    assets=figures(destination)
    md,docx=destination/'final_report_ati.md',destination/'final_report_ati.docx'
    md.write_text(text);render_word(text,docx,destination)
    for path,expected in inputs.items():
        if checksum(path)!=expected:raise ValueError(f'Report input changed during build: {path}')
    result=dict(template_used=str(TEMPLATE.relative_to(ROOT)),submission_ready=False,model_inference_performed=False,
                body_pages=None,body_page_check='Requires document-layout rendering',
                missing_fields=['authors','group_name','course_section','representative','contribution_percentages',
                                'generative_ai_declaration','private_asset_links','final_executed_notebook','submission_prefix'],
                input_sha256={str(Path(p).relative_to(ROOT)):value for p,value in inputs.items()},
                generated_sha256={p.name:checksum(p) for p in (md,docx)},
                figure_sha256={str(p.relative_to(destination)):checksum(p) for p in assets})
    (destination/'final_report_ati_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def cache_toc_from_pdf(destination, pdf_path):
    """Cache the native TOC using page numbers from an actual layout rendering.

    The main section restarts at page 1, so expanding front matter cannot change
    its page numbers. Word can refresh the TOC/PAGEREF fields after later edits.
    """
    destination, pdf_path = Path(destination), Path(pdf_path)
    path = destination/'final_report_ati.docx'
    document = Document(path)
    pages = [page.extract_text() for page in PdfReader(pdf_path).pages]
    normalize = lambda text: ' '.join(text.split())
    page_lines = [{normalize(line) for line in text.splitlines()} for text in pages]
    headings = [p for p in document.paragraphs if p.style is not None and p.style.style_id in ('Heading1','Heading2')]
    locations = []
    for paragraph in headings:
        matches = [i for i,lines in enumerate(page_lines) if normalize(paragraph.text) in lines]
        if len(matches) != 1:
            raise ValueError(f'Heading must occur once in the rendered PDF: {paragraph.text}')
        locations.append(matches[0])
    intro = next(i for p,i in zip(headings,locations) if p.text.startswith('1  Introduction'))
    references = next(i for p,i in zip(headings,locations) if p.text=='References')
    if not 15 <= references-intro <= 20:
        raise ValueError('Rendered Sections 1–7 must occupy 15–20 pages')
    toc = next(e for e in document.element.body if e.tag==qn('w:sdt'))
    content = toc.find(qn('w:sdtContent'))
    for element in list(content)[1:-1]:content.remove(element)
    bookmark_id = 1000
    for paragraph,location in zip(headings,locations):
        bookmark_id += 1; name=f'_ATIHeading{bookmark_id}'
        start=OxmlElement('w:bookmarkStart');start.set(qn('w:id'),str(bookmark_id));start.set(qn('w:name'),name)
        end=OxmlElement('w:bookmarkEnd');end.set(qn('w:id'),str(bookmark_id))
        paragraph._p.insert(1 if paragraph._p.pPr is not None else 0,start);paragraph._p.append(end)
        entry=document.add_paragraph()
        entry.paragraph_format.keep_with_next=False
        entry.paragraph_format.space_after=Pt(0)
        if paragraph.style.style_id=='Heading2':entry.paragraph_format.left_indent=Cm(.5)
        entry.paragraph_format.tab_stops.add_tab_stop(Cm(15.9),WD_TAB_ALIGNMENT.RIGHT)
        link=OxmlElement('w:hyperlink');link.set(qn('w:anchor'),name)
        run=OxmlElement('w:r');text=OxmlElement('w:t');text.text=paragraph.text;run.append(text);link.append(run)
        entry._p.append(link);entry.add_run('\t')
        field(entry,f'PAGEREF {name} \\h',location-intro+1)
        content.insert(len(content)-1,entry._p)
    document.save(path)
    proof_path=destination/'final_report_ati_verification.json'
    proof=read_json(proof_path)
    proof.update(body_pages=references-intro,body_page_check='15–20 pages verified in a LibreOffice PDF rendering',
                 toc_cached=True,toc_heading_count=len(headings),
                 toc_layout_source_sha256=checksum(pdf_path))
    proof['generated_sha256'][path.name]=checksum(path)
    proof_path.write_text(json.dumps(proof,indent=2)+'\n')
    return proof


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,default=ROOT/'reports')
    parser.add_argument('--cache-toc-from-pdf',type=Path,help='PDF rendered from this report before caching the TOC')
    args=parser.parse_args();result=build(args.output_dir)
    if args.cache_toc_from_pdf:result=cache_toc_from_pdf(args.output_dir,args.cache_toc_from_pdf)
    print(json.dumps({k:result[k] for k in ('template_used','submission_ready','body_page_check','missing_fields')},indent=2))
