"""Final manuscript numbers and PDF export, without training or inference."""
from pathlib import Path
import hashlib
import sys
import tempfile
import unittest
import zipfile
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
from pypdf import PdfReader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from build_final_paper import comparison_table, markdown_table, fill_template, render_pdf, render_docx, build


class FinalPaperTests(unittest.TestCase):
    def test_main_table_uses_all28_and_keeps_three_decoders(self):
        rows = []
        for scope, score in [('all28', .4), ('emotion27', .9)]:
            for strategy in ('fixed', 'global', 'per_label'):
                rows.append(dict(scope=scope, strategy=strategy, macro_f1=score, micro_f1=score,
                                 hamming_loss=.1, lrap=.7))
        frame = pd.DataFrame(rows)
        result = comparison_table(frame, frame)
        self.assertIn('0.400000', result)
        self.assertNotIn('0.900000', result)
        for label in ('Fixed 0.5', 'Global', 'Per-label'):
            self.assertIn(label, result)

    def test_empty_metrics_are_na_and_not_invented_zero(self):
        result = markdown_table(pd.DataFrame({'n': [0], 'Macro-F1': [np.nan]}))
        self.assertIn('N/A', result)
        self.assertNotIn('nan', result)

    def test_unknown_template_field_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'template'):
            fill_template('{{known}} {{missing}}', {'known': 'verified'})

    def test_numeric_scope_names_are_supported_in_template_fields(self):
        self.assertEqual(fill_template('{{emotion27_results}}', {'emotion27_results': 'verified projection'}), 'verified projection')

    def test_pdf_exports_unicode_links_tables_and_readable_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / 'paper.pdf'
            source = '# Threshold transfer\n\n## Abstract\n\n**Macro-F1** uses τ ≥ 0.5. [Source](https://example.org/paper).\n\n| Metric | Value |\n| --- | --- |\n| Test | 0.591254 |\n'
            render_pdf(source, destination, Path(tmp))
            reader = PdfReader(destination)
            text = '\n'.join(p.extract_text() or '' for p in reader.pages)
            self.assertIn('0.591254', text)
            self.assertIn('Abstract', text)
            self.assertIn('τ', text)
            self.assertNotIn('**', text)
            annotations = [a.get_object() for p in reader.pages for a in p.get('/Annots', [])]
            self.assertTrue(any(a.get('/A', {}).get('/URI') == 'https://example.org/paper' for a in annotations))

    def test_complete_paper_build_preserves_frozen_inputs(self):
        root = Path(__file__).resolve().parents[1]
        protected = [*list((root / 'reports/final28/predictions').glob('*.npz')),
                     root / 'reports/final28/thresholds.json', root / 'reports/final28/pre_test_lock.json',
                     root / 'docs/final_protocol.md', root / 'configs/final_thresholds.json']
        before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
        with tempfile.TemporaryDirectory() as tmp:
            result = build(Path(tmp))
            manuscript = (Path(tmp) / 'final_report.md').read_text()
            self.assertNotIn('{{', manuscript)
            self.assertTrue(result['content_complete'])
            self.assertFalse(result['phobert_inference_performed'])
            for name in ('Abstract', 'Introduction', 'Data', 'Method', 'Experiments', 'Results',
                         'Discussion', 'Limitations', 'Conclusion', 'References'):
                self.assertIn(name, manuscript)
            self.assertGreater(len(PdfReader(Path(tmp) / 'final_report.pdf').pages), 1)
            with zipfile.ZipFile(Path(tmp) / 'final_report.docx') as archive:
                ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
                document = ET.fromstring(archive.read('word/document.xml'))
                word_text = '\n'.join(document.itertext())
                for value in ('Qualitative Error Analysis', '0.591254', '0.585415', 'Appendix B.'):
                    self.assertIn(value, word_text)
                self.assertEqual(len(document.findall('.//w:tbl', ns)), 10)
                self.assertEqual(len(document.findall('.//w:drawing', ns)), 3)
                self.assertNotIn('AI-assisted', word_text)
            self.assertEqual(result['generated_sha256']['final_report.docx'],
                             hashlib.sha256((Path(tmp) / 'final_report.docx').read_bytes()).hexdigest())
        self.assertEqual(before, {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in protected})

    def test_docx_keeps_editable_tables_links_images_and_page_breaks(self):
        from docx import Document
        root = Path(__file__).resolve().parents[1]
        source = ('# Threshold transfer\n\n## Abstract\n\n**Macro-F1** uses τ ≥ 0.5. '
                  '[Source](https://example.org/paper).\n\n| Metric | Value |\n| --- | --- |\n'
                  '| Test | 0.591254 |\n\n![Figure 1. Comparison](final28/figures/validation_test.png)'
                  '\n\n<!-- pagebreak -->\n\n## Appendix\n\n`report`\n')
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / 'paper.docx'
            render_docx(source, destination, root / 'reports')
            document = Document(destination)
            self.assertEqual(document.tables[0].cell(1, 1).text, '0.591254')
            self.assertEqual(len(document.inline_shapes), 1)
            self.assertLessEqual(document.inline_shapes[0].width,
                                 document.sections[0].page_width - document.sections[0].left_margin - document.sections[0].right_margin)
            self.assertTrue(any(run.bold for p in document.paragraphs for run in p.runs if run.text == 'Macro-F1'))
            with zipfile.ZipFile(destination) as archive:
                self.assertIn('https://example.org/paper', archive.read('word/_rels/document.xml.rels').decode())
                xml = archive.read('word/document.xml').decode()
                self.assertIn('w:type="page"', xml)
                self.assertIn('w:tblHeader', xml)


if __name__ == '__main__':
    unittest.main()
