""" Text-based normalizers, used to mitigate simple attacks against watermarking.

This implementation is unlikely to be a complete list of all possible exploits within the unicode standard,
it represents our best effort at the time of writing.

These normalizers can be used as stand-alone normalizers. They could be made to conform to HF tokenizers standard, but that would
require messing with the limited rust interface of tokenizers.NormalizedString
"""
from collections import defaultdict
from functools import cache

import re
import unicodedata

"""Updated version of core.py from
https://github.com/yamatt/homoglyphs/tree/main/homoglyphs_fork
for modern python3
"""

from collections import defaultdict
import json
from itertools import product
import os
import unicodedata

# Actions if char not in alphabet
STRATEGY_LOAD = 1  # load category for this char
STRATEGY_IGNORE = 2  # add char to result
STRATEGY_REMOVE = 3  # remove char from result

ASCII_RANGE = range(128)


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_LOCATION = os.path.join(CURRENT_DIR, "homoglyph_data")


class Categories:
    """
    Work with aliases from ISO 15924.
    https://en.wikipedia.org/wiki/ISO_15924#List_of_codes
    """

    fpath = os.path.join(DATA_LOCATION, "categories.json")

    @classmethod
    def _get_ranges(cls, categories):
        """
        :return: iter: (start code, end code)
        :rtype: list
        """
        with open(cls.fpath, encoding="utf-8") as f:
            data = json.load(f)

        for category in categories:
            if category not in data["aliases"]:
                raise ValueError("Invalid category: {}".format(category))

        for point in data["points"]:
            if point[2] in categories:
                yield point[:2]

    @classmethod
    def get_alphabet(cls, categories):
        """
        :return: set of chars in alphabet by categories list
        :rtype: set
        """
        alphabet = set()
        for start, end in cls._get_ranges(categories):
            chars = (chr(code) for code in range(start, end + 1))
            alphabet.update(chars)
        return alphabet

    @classmethod
    def detect(cls, char):
        """
        :return: category
        :rtype: str
        """
        with open(cls.fpath, encoding="utf-8") as f:
            data = json.load(f)

        # try detect category by unicodedata
        try:
            category = unicodedata.name(char).split()[0]
        except (TypeError, ValueError):
            # In Python2 unicodedata.name raise error for non-unicode chars
            # Python3 raise ValueError for non-unicode characters
            pass
        else:
            if category in data["aliases"]:
                return category

        # try detect category by ranges from JSON file.
        code = ord(char)
        for point in data["points"]:
            if point[0] <= code <= point[1]:
                return point[2]

    @classmethod
    def get_all(cls):
        with open(cls.fpath, encoding="utf-8") as f:
            data = json.load(f)
        return set(data["aliases"])


class Languages:
    fpath = os.path.join(DATA_LOCATION, "languages.json")

    @classmethod
    def get_alphabet(cls, languages):
        """
        :return: set of chars in alphabet by languages list
        :rtype: set
        """
        with open(cls.fpath, encoding="utf-8") as f:
            data = json.load(f)
        alphabet = set()
        for lang in languages:
            if lang not in data:
                raise ValueError("Invalid language code: {}".format(lang))
            alphabet.update(data[lang])
        return alphabet

    @classmethod
    def detect(cls, char):
        """
        :return: set of languages which alphabet contains passed char.
        :rtype: set
        """
        with open(cls.fpath, encoding="utf-8") as f:
            data = json.load(f)
        languages = set()
        for lang, alphabet in data.items():
            if char in alphabet:
                languages.add(lang)
        return languages

    @classmethod
    def get_all(cls):
        with open(cls.fpath, encoding="utf-8") as f:
            data = json.load(f)
        return set(data.keys())


class Homoglyphs:
    def __init__(
        self,
        categories=None,
        languages=None,
        alphabet=None,
        strategy=STRATEGY_IGNORE,
        ascii_strategy=STRATEGY_IGNORE,
        ascii_range=ASCII_RANGE,
    ):
        # strategies
        if strategy not in (STRATEGY_LOAD, STRATEGY_IGNORE, STRATEGY_REMOVE):
            raise ValueError("Invalid strategy")
        self.strategy = strategy
        self.ascii_strategy = ascii_strategy
        self.ascii_range = ascii_range

        # Homoglyphs must be initialized by any alphabet for correct work
        if not categories and not languages and not alphabet:
            categories = ("LATIN", "COMMON")

        # cats and langs
        self.categories = set(categories or [])
        self.languages = set(languages or [])

        # alphabet
        self.alphabet = set(alphabet or [])
        if self.categories:
            alphabet = Categories.get_alphabet(self.categories)
            self.alphabet.update(alphabet)
        if self.languages:
            alphabet = Languages.get_alphabet(self.languages)
            self.alphabet.update(alphabet)
        self.table = self.get_table(self.alphabet)

    @staticmethod
    def get_table(alphabet):
        table = defaultdict(set)
        with open(os.path.join(DATA_LOCATION, "confusables_sept2022.json")) as f:
            data = json.load(f)
        for char in alphabet:
            if char in data:
                for homoglyph in data[char]:
                    if homoglyph in alphabet:
                        table[char].add(homoglyph)
        return table

    @staticmethod
    def get_restricted_table(source_alphabet, target_alphabet):
        table = defaultdict(set)
        with open(os.path.join(DATA_LOCATION, "confusables_sept2022.json")) as f:
            data = json.load(f)
        for char in source_alphabet:
            if char in data:
                for homoglyph in data[char]:
                    if homoglyph in target_alphabet:
                        table[char].add(homoglyph)
        return table

    @staticmethod
    def uniq_and_sort(data):
        result = list(set(data))
        result.sort(key=lambda x: (-len(x), x))
        return result

    def _update_alphabet(self, char):
        # try detect languages
        langs = Languages.detect(char)
        if langs:
            self.languages.update(langs)
            alphabet = Languages.get_alphabet(langs)
            self.alphabet.update(alphabet)
        else:
            # try detect categories
            category = Categories.detect(char)
            if category is None:
                return False
            self.categories.add(category)
            alphabet = Categories.get_alphabet([category])
            self.alphabet.update(alphabet)
        # update table for new alphabet
        self.table = self.get_table(self.alphabet)
        return True

    def _get_char_variants(self, char):
        if char not in self.alphabet:
            if self.strategy == STRATEGY_LOAD:
                if not self._update_alphabet(char):
                    return []
            elif self.strategy == STRATEGY_IGNORE:
                return [char]
            elif self.strategy == STRATEGY_REMOVE:
                return []

        # find alternative chars for current char
        alt_chars = self.table.get(char, set())
        if alt_chars:
            # find alternative chars for alternative chars for current char
            alt_chars2 = [self.table.get(alt_char, set()) for alt_char in alt_chars]
            # combine all alternatives
            alt_chars.update(*alt_chars2)
        # add current char to alternatives
        alt_chars.add(char)

        # uniq, sort and return
        return self.uniq_and_sort(alt_chars)

    def _get_combinations(self, text, ascii=False):
        variations = []
        for char in text:
            alt_chars = self._get_char_variants(char)

            if ascii:
                alt_chars = [char for char in alt_chars if ord(char) in self.ascii_range]
                if not alt_chars and self.ascii_strategy == STRATEGY_IGNORE:
                    return

            if alt_chars:
                variations.append(alt_chars)
        if variations:
            for variant in product(*variations):
                yield "".join(variant)

    def get_combinations(self, text):
        return list(self._get_combinations(text))

    def _to_ascii(self, text):
        for variant in self._get_combinations(text, ascii=True):
            if max(map(ord, variant)) in self.ascii_range:
                yield variant

    def to_ascii(self, text):
        return self.uniq_and_sort(self._to_ascii(text))
def normalization_strategy_lookup(strategy_name: str) -> object:
    if strategy_name == "unicode":
        return UnicodeSanitizer()
    elif strategy_name == "homoglyphs":
        return HomoglyphCanonizer()
    elif strategy_name == "truecase":
        return TrueCaser()


class HomoglyphCanonizer:
    """Attempts to detect homoglyph attacks and find a consistent canon.

    This function does so on a per-ISO-category level. Language-level would also be possible (see commented code).
    """

    def __init__(self):
        self.homoglyphs = None

    def __call__(self, homoglyphed_str: str) -> str:
        # find canon:
        target_category, all_categories = self._categorize_text(homoglyphed_str)
        homoglyph_table = self._select_canon_category_and_load(target_category, all_categories)
        return self._sanitize_text(target_category, homoglyph_table, homoglyphed_str)

    def _categorize_text(self, text: str) -> dict:
        iso_categories = defaultdict(int)
        # self.iso_languages = defaultdict(int)

        for char in text:
            iso_categories[hg.Categories.detect(char)] += 1
            # for lang in hg.Languages.detect(char):
            #     self.iso_languages[lang] += 1
        target_category = max(iso_categories, key=iso_categories.get)
        all_categories = tuple(iso_categories)
        return target_category, all_categories

    @cache
    def _select_canon_category_and_load(
        self, target_category: str, all_categories: tuple[str]
    ) -> dict:
        homoglyph_table = hg.Homoglyphs(
            categories=(target_category, "COMMON")
        )  # alphabet loaded here from file

        source_alphabet = hg.Categories.get_alphabet(all_categories)
        restricted_table = homoglyph_table.get_restricted_table(
            source_alphabet, homoglyph_table.alphabet
        )  # table loaded here from file
        return restricted_table

    def _sanitize_text(
        self, target_category: str, homoglyph_table: dict, homoglyphed_str: str
    ) -> str:
        sanitized_text = ""
        for char in homoglyphed_str:
            # langs = hg.Languages.detect(char)
            cat = hg.Categories.detect(char)
            if target_category in cat or "COMMON" in cat or len(cat) == 0:
                sanitized_text += char
            else:
                sanitized_text += list(homoglyph_table[char])[0]
        return sanitized_text


class UnicodeSanitizer:
    """Regex-based unicode sanitzer. Has different levels of granularity.

    * ruleset="whitespaces"    - attempts to remove only whitespace unicode characters
    * ruleset="IDN.blacklist"  - does its best to remove unusual unicode based on  Network.IDN.blacklist characters
    * ruleset="ascii"          - brute-forces all text into ascii

    This is unlikely to be a comprehensive list.

    You can find a more comprehensive discussion at https://www.unicode.org/reports/tr36/
    and https://www.unicode.org/faq/security.html
    """

    def __init__(self, ruleset="whitespaces"):
        if ruleset == "whitespaces":
            """Documentation:
            \u00A0: Non-breaking space
            \u1680: Ogham space mark
            \u180E: Mongolian vowel separator
            \u2000-\u200B: Various space characters, including en space, em space, thin space, hair space, zero-width space, and zero-width non-joiner
            \u200C\u200D: Zero-width non-joiner and zero-width joiner
            \u200E,\u200F: Left-to-right-mark, Right-to-left-mark
            \u2060: Word joiner
            \u2063: Invisible separator
            \u202F: Narrow non-breaking space
            \u205F: Medium mathematical space
            \u3000: Ideographic space
            \uFEFF: Zero-width non-breaking space
            \uFFA0: Halfwidth hangul filler
            \uFFF9\uFFFA\uFFFB: Interlinear annotation characters
            \uFE00-\uFE0F: Variation selectors
            \u202A-\u202F: Embedding characters
            \u3164: Korean hangul filler.

            Note that these characters are not always superfluous whitespace characters!
            """

            self.pattern = re.compile(
                r"[\u00A0\u1680\u180E\u2000-\u200B\u200C\u200D\u200E\u200F\u2060\u2063\u202F\u205F\u3000\uFEFF\uFFA0\uFFF9\uFFFA\uFFFB"
                r"\uFE00\uFE01\uFE02\uFE03\uFE04\uFE05\uFE06\uFE07\uFE08\uFE09\uFE0A\uFE0B\uFE0C\uFE0D\uFE0E\uFE0F\u3164\u202A\u202B\u202C\u202D"
                r"\u202E\u202F]"
            )
        elif ruleset == "IDN.blacklist":
            """Documentation:
            [\u00A0\u1680\u180E\u2000-\u200B\u202F\u205F\u2060\u2063\uFEFF]: Matches any whitespace characters in the Unicode character
                        set that are included in the IDN blacklist.
            \uFFF9-\uFFFB: Matches characters that are not defined in Unicode but are used as language tags in various legacy encodings.
                        These characters are not allowed in domain names.
            \uD800-\uDB7F: Matches the first part of a surrogate pair. Surrogate pairs are used to represent characters in the Unicode character
                        set that cannot be represented by a single 16-bit value. The first part of a surrogate pair is in the range U+D800 to U+DBFF,
                        and the second part is in the range U+DC00 to U+DFFF.
            \uDB80-\uDBFF][\uDC00-\uDFFF]?: Matches the second part of a surrogate pair. The second part of a surrogate pair is in the range U+DC00
                        to U+DFFF, and is optional.
            [\uDB40\uDC20-\uDB40\uDC7F][\uDC00-\uDFFF]: Matches certain invalid UTF-16 sequences which should not appear in IDNs.
            """

            self.pattern = re.compile(
                r"[\u00A0\u1680\u180E\u2000-\u200B\u202F\u205F\u2060\u2063\uFEFF\uFFF9-\uFFFB\uD800-\uDB7F\uDB80-\uDBFF]"
                r"[\uDC00-\uDFFF]?|[\uDB40\uDC20-\uDB40\uDC7F][\uDC00-\uDFFF]"
            )
        else:
            """Documentation:
            This is a simple restriction to "no-unicode", using only ascii characters. Control characters are included.
            """
            self.pattern = re.compile(r"[^\x00-\x7F]+")

    def __call__(self, text: str) -> str:
        text = unicodedata.normalize("NFC", text)  # canon forms
        text = self.pattern.sub(" ", text)  # pattern match
        text = re.sub(" +", " ", text)  # collapse whitespaces
        text = "".join(
            c for c in text if unicodedata.category(c) != "Cc"
        )  # Remove any remaining non-printable characters
        return text


class TrueCaser:
    """True-casing, is a capitalization normalization that returns text to its original capitalization.

    This defends against attacks that wRIte TeXt lIkE spOngBoB.

    Here, a simple POS-tagger is used.
    """

    uppercase_pos = ["PROPN"]  # Name POS tags that should be upper-cased

    def __init__(self, backend="spacy"):
        if backend == "spacy":
            import spacy

            self.nlp = spacy.load("en_core_web_sm")
            self.normalize_fn = self._spacy_truecasing
        else:
            from nltk import pos_tag, word_tokenize  # noqa
            import nltk

            nltk.download("punkt")
            nltk.download("averaged_perceptron_tagger")
            nltk.download("universal_tagset")
            self.normalize_fn = self._nltk_truecasing

    def __call__(self, random_capitalized_string: str) -> str:
        truecased_str = self.normalize_fn(random_capitalized_string)
        return truecased_str

    def _spacy_truecasing(self, random_capitalized_string: str):
        doc = self.nlp(random_capitalized_string.lower())
        POS = self.uppercase_pos
        truecased_str = "".join(
            [
                w.text_with_ws.capitalize() if w.pos_ in POS or w.is_sent_start else w.text_with_ws
                for w in doc
            ]
        )
        return truecased_str

    def _nltk_truecasing(self, random_capitalized_string: str):
        from nltk import pos_tag, word_tokenize
        import nltk

        nltk.download("punkt")
        nltk.download("averaged_perceptron_tagger")
        nltk.download("universal_tagset")
        POS = ["NNP", "NNPS"]

        tagged_text = pos_tag(word_tokenize(random_capitalized_string.lower()))
        truecased_str = " ".join([w.capitalize() if p in POS else w for (w, p) in tagged_text])
        return truecased_str