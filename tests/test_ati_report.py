"""The submission report must use the supplied template without altering evidence."""
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

from docx import Document
from docx.oxml.ns import qn
from lxml import etree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from build_ati_report import build, report_values

ROOT = Path(__file__).resolve().parents[1]


class ATIReportTests(unittest.TestCase):
    def test_numeric_content_keeps_baseline_and_final_scopes_separate(self):
        values = report_values()
        self.assertIn('0.597410', values['baseline_results'])
        self.assertIn('0.578341', values['baseline_results'])
        self.assertIn('0.589964', values['primary_results'])
        self.assertIn('0.591254', values['primary_results'])
        self.assertIn('0.585415', values['primary_results'])
        self.assertNotIn('0.597410', values['primary_results'])
        self.assertEqual(values['rq4_original_macro'], '0.589964')
        self.assertEqual(values['rq4_diacritics_macro'], '0.329749')
        self.assertEqual(values['rq4_typo_macro'], '0.550578')
        self.assertEqual(values['rq4_diacritics_loss'], '26.02')
        self.assertEqual(values['rq4_typo_loss'], '3.94')
        for sample in ('you003220', 'tik017852', 'you001985', '247', '18169'):
            self.assertIn(sample, values['failure_examples'])

    def test_template_sections_fields_and_unknown_identity_are_preserved(self):
        protected = [ROOT/'docs/ATI_Final_Report_Template.docx', ROOT/'reports/final_report.docx',
                     ROOT/'reports/final28/thresholds.json', ROOT/'reports/final28/pre_test_lock.json']
        before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
        with tempfile.TemporaryDirectory() as tmp:
            result = build(Path(tmp))
            path = Path(tmp)/'final_report_ati.docx'
            doc = Document(path)
            style = next(s for s in doc.styles if s.style_id == 'Normal')
            self.assertEqual(style.font.name, 'Times New Roman')
            self.assertEqual(style.font.size.pt, 12)
            self.assertEqual(style.paragraph_format.line_spacing, 1.5)
            self.assertEqual(len(doc.sections), 3)
            for section in doc.sections:
                self.assertAlmostEqual(section.left_margin.cm, 2.5, delta=.02)
                self.assertAlmostEqual(section.right_margin.cm, 2.5, delta=.02)
            headings = [p.text for p in doc.paragraphs if p.style is not None and p.style.style_id == 'Heading1']
            main = [h for h in headings if h[:1].isdigit()]
            self.assertEqual([h.split()[0] for h in main], list('1234567'))
            template = Document(protected[0])
            expected_subsections = [p.text for p in template.paragraphs
                                    if p.style and p.style.style_id == 'Heading2' and p.text[:1].isdigit()]
            actual_subsections = [p.text for p in doc.paragraphs
                                  if p.style and p.style.style_id == 'Heading2' and p.text[:1].isdigit()]
            self.assertEqual(actual_subsections, expected_subsections)
            self.assertEqual(sum(p.style and p.style.style_id == 'ListParagraph' and bool(p.text)
                                 for p in doc.paragraphs if p.style), 3)
            headers = [tuple(c.text for c in t.rows[0].cells) for t in doc.tables]
            self.assertEqual(headers[2:6], [
                ('RQ', 'Question', 'Hypothesis', 'Control (Midterm)'),
                ('Split / domain', 'Samples', 'Classes or labels', 'Notes'),
                ('Model', 'Checkpoint', 'Input size', 'Epochs', 'Batch', 'LR', 'Params'),
                ('Model', 'Fixed 0.5 (A)', 'Tuned cutoff (B)', 'Δ (B − A)'),
            ])
            self.assertEqual(headers[-2:], [
                ('Student ID', 'Full name', 'Main tasks', '%'),
                ('Tool', 'Purpose', 'Sections or files affected'),
            ])
            captions = [p.text for p in doc.paragraphs if p.style and p.style.style_id == 'Caption']
            self.assertEqual([c.split(':')[0] for c in captions if c.startswith(('Table 1:', 'Table 2:', 'Table 3:', 'Table 4:', 'Table 5:'))],
                             [f'Table {i}' for i in range(1, 6)])
            for label in ('Appendix A', 'Appendix B', 'Appendix C'):
                self.assertTrue(any(h.startswith(label) for h in headings))
            all_text='\n'.join(p.text for p in doc.paragraphs)+'\n'+'\n'.join(c.text for t in doc.tables for r in t.rows for c in r.cells)
            self.assertNotIn('Template Instructions', all_text)
            self.assertNotIn('Recommended length', all_text)
            self.assertIn('[Full name]', all_text)
            self.assertIn('[Student ID]', all_text)
            self.assertIn('P02', all_text)
            self.assertIn('[PREFIX]', '\n'.join(p.text for s in doc.sections for p in s.header.paragraphs))
            with zipfile.ZipFile(path) as archive:
                xml=etree.fromstring(archive.read('word/document.xml'))
                ns={'w':qn('w:p').split('}')[0][1:], 'm':'http://schemas.openxmlformats.org/officeDocument/2006/math'}
                instructions=xml.xpath('//w:instrText/text()',namespaces=ns)
                self.assertTrue(any('TOC' in text for text in instructions))
                self.assertGreaterEqual(sum('SEQ Table' in text for text in instructions), 10)
                self.assertGreaterEqual(sum('SEQ Figure' in text for text in instructions), 3)
                self.assertGreaterEqual(len(xml.xpath('//m:oMath',namespaces=ns)), 4)
            self.assertFalse(result['submission_ready'])
            self.assertFalse(result['model_inference_performed'])
            self.assertIn('authors', result['missing_fields'])
            self.assertIn('rq4_executed_notebook', result['missing_fields'])
            self.assertIn('reports/rq4_noise/test/tables/comparison.csv', result['input_sha256'])
            self.assertTrue(any(c.startswith('Figure 3: Change in all28 test Macro-F1') for c in captions))
            self.assertGreaterEqual(len(doc.inline_shapes), 4)
        self.assertEqual(before, {p:hashlib.sha256(p.read_bytes()).hexdigest() for p in protected})


if __name__ == '__main__':
    unittest.main()
