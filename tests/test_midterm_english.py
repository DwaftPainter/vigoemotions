"""Translation must preserve the archived Midterm evidence and numeric tables."""
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

from docx import Document
from pypdf import PdfReader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from build_midterm_english import build, translated_markdown, verify_translation

ROOT = Path(__file__).resolve().parents[1]


class MidtermEnglishTests(unittest.TestCase):
    def test_full_translation_keeps_tables_figures_and_historical_scope(self):
        source = (ROOT / 'reports/midterm_report_vi.md').read_text()
        english = translated_markdown(source)
        verify_translation(source, english)
        self.assertIn('## 6. Answers to the Five Midterm Questions', english)
        self.assertIn('### Q5.', english)
        self.assertIn('### A.28.', english)
        self.assertIn('Test performance had not been evaluated at the Midterm stage', english)
        self.assertNotIn('Không áp dụng', english)
        self.assertNotIn('AI-assisted', english)

    def test_changed_metric_or_missing_figure_is_rejected(self):
        source = (ROOT / 'reports/midterm_report_vi.md').read_text()
        english = translated_markdown(source)
        with self.assertRaisesRegex(ValueError, 'table'):
            verify_translation(source, english.replace('| PhoBERT | 0.5974 |', '| PhoBERT | 0.9999 |', 1))
        with self.assertRaisesRegex(ValueError, 'figure'):
            verify_translation(source, english.replace('figures/label_frequency.png', 'missing.png'))

    def test_exports_keep_original_vietnamese_and_completed_studies_unchanged(self):
        protected = [ROOT / p for p in ('reports/midterm_report_vi.md', 'reports/midterm_report_vi.pdf',
                     'reports/midterm_slides.pdf', 'reports/final_report.pdf', 'reports/final_report.docx')]
        before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
        with tempfile.TemporaryDirectory() as tmp:
            result = build(Path(tmp))
            word = Document(Path(tmp) / 'midterm_report.docx')
            self.assertEqual(len(word.tables), 13)
            self.assertEqual(len(word.inline_shapes), 5)
            self.assertEqual(len(word.tables[-2].rows), 28)
            self.assertEqual(len(word.tables[-1].rows), 29)
            reader = PdfReader(Path(tmp) / 'midterm_report.pdf')
            self.assertIn('Midterm Report', reader.pages[0].extract_text())
            self.assertIn('Midterm', word.sections[0].footer.paragraphs[0].text)
            self.assertNotIn('ATI Final', word.sections[0].footer.paragraphs[0].text)
            self.assertEqual(result['numeric_tables_preserved'], 12)
            self.assertEqual(result['translated_review_cases'], 6)
            for name, sha in result['generated_sha256'].items():
                self.assertEqual(hashlib.sha256((Path(tmp) / name).read_bytes()).hexdigest(), sha)
        self.assertEqual(before, {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in protected})


if __name__ == '__main__':
    unittest.main()
