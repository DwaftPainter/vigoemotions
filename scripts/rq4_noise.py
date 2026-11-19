"""Deterministic text perturbations applied after RQ1 normalization."""
import hashlib
import unicodedata

CONDITIONS = ('original', 'no_diacritics', 'qwerty_typo')
VIETNAMESE_VOWELS = (
    'àáảãạăằắẳẵặâầấẩẫậ', 'èéẻẽẹêềếểễệ', 'ìíỉĩị',
    'òóỏõọôồốổỗộơờớởỡợ', 'ùúủũụưừứửữự', 'ỳýỷỹỵ',
)
MARKS = {char: base for base, variants in zip('aeiouy', VIETNAMESE_VOWELS) for char in variants}
MARKS['đ'] = 'd'
DIACRITICS = str.maketrans({**MARKS, **{key.upper(): value.upper() for key, value in MARKS.items()}})
KEYS = {char: (column + offset, row)
        for row, (offset, keys) in enumerate(((0, 'qwertyuiop'), (.25, 'asdfghjkl'), (.75, 'zxcvbnm')))
        for column, char in enumerate(keys)}
NEIGHBORS = {
    char: tuple(sorted(other for other, (ox, oy) in KEYS.items()
                       if 0 < (x - ox)**2 + (y - oy)**2 <= 1.25**2))
    for char, (x, y) in KEYS.items()
}


def remove_diacritics(text):
    """Remove Vietnamese vowel marks and map đ/Đ to d/D."""
    return unicodedata.normalize('NFC', text).translate(DIACRITICS)


def is_typo_eligible(char):
    return char.isascii() and char.lower() in NEIGHBORS


def qwerty_typos(text, *, sample_id, seed, rate):
    """Replace eligible characters independently, reproducibly across input order."""
    if not 0 <= rate <= 1:
        raise ValueError('Typo rate must be in [0, 1]')
    cutoff = int(rate * 2**64)
    characters = list(text)
    for index, char in enumerate(characters):
        if not is_typo_eligible(char):
            continue
        draw = hashlib.sha256(f'{seed}:{index}:{sample_id}'.encode('utf-8')).digest()
        if int.from_bytes(draw[:8], 'big') < cutoff:
            options = NEIGHBORS[char.lower()]
            replacement = options[int.from_bytes(draw[8:16], 'big') % len(options)]
            characters[index] = replacement.upper() if char.isupper() else replacement
    return ''.join(characters)


def prepare_conditions(frame, *, seed, rate):
    """Keep split IDs and targets in order and log coverage without exporting text."""
    original = frame.normalized.tolist()
    variants = {
        'original': original,
        'no_diacritics': [remove_diacritics(text) for text in original],
        'qwerty_typo': [qwerty_typos(text, sample_id=sample_id, seed=seed, rate=rate)
                        for sample_id, text in zip(frame.id, original, strict=True)],
    }
    eligible_diacritics = sum(ord(char) in DIACRITICS for text in original for char in text)
    eligible_typos = sum(is_typo_eligible(char) for text in original for char in text)
    eligible = {'original': 0, 'no_diacritics': eligible_diacritics, 'qwerty_typo': eligible_typos}
    counts = {}
    for condition, transformed in variants.items():
        counts[condition] = {
            'samples': len(original),
            'changed_samples': sum(a != b for a, b in zip(original, transformed, strict=True)),
            'eligible_characters': eligible[condition],
            'edited_characters': sum(a != b for before, after in zip(original, transformed, strict=True)
                                     for a, b in zip(before, after, strict=True)),
        }
    return {name: frame.assign(normalized=texts) for name, texts in variants.items()}, counts
