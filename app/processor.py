import re
import logging

logger = logging.getLogger(__name__)

class TextProcessor:
    def __init__(self, max_chars: int = 200):
        self.max_chars = max_chars

    def _split_into_sentences(self, text: str):
        """
        Your robust logic: handles dialogue, abbreviations,
        decimals, and complex punctuation.
        """
        if not text or not text.strip():
            return []

        text = re.sub(r'\s+', ' ', text.strip())

        abbreviations = {
            'Mr.', 'Mrs.', 'Ms.', 'Dr.', 'Prof.', 'Sr.', 'Jr.',
            'vs.', 'etc.', 'i.e.', 'e.g.', 'cf.', 'al.', 'Co.',
            'Corp.', 'Inc.', 'Ltd.', 'St.', 'Ave.', 'Blvd.',
            'U.S.', 'U.K.', 'U.N.', 'E.U.', 'Ph.D.', 'M.D.',
            'B.A.', 'M.A.', 'a.m.', 'p.m.', 'A.M.', 'P.M.'
        }

        protected_abbrevs = {}
        for i, abbrev in enumerate(abbreviations):
            placeholder = f"__ABBREV_{i}__"
            if abbrev in text:
                text = text.replace(abbrev, placeholder)
                protected_abbrevs[placeholder] = abbrev

        decimal_pattern = r'\b\d+\.\d+\b'
        decimals = re.findall(decimal_pattern, text)
        for i, decimal in enumerate(decimals):
            placeholder = f"__DECIMAL_{i}__"
            text = text.replace(decimal, placeholder)
            protected_abbrevs[placeholder] = decimal

        sentences = []
        current_sentence = ""
        i = 0

        while i < len(text):
            char = text[i]
            current_sentence += char

            if (i < len(text) - 2 and char.islower() and text[i + 1] == ' ' and
                text[i + 2].isupper() and len(current_sentence.split()) > 3):
                remaining = text[i + 2:]
                next_word_match = re.match(r'^[A-Z][a-z]+', remaining)
                if next_word_match:
                    next_word = next_word_match.group()
                    if next_word not in ['The', 'This', 'That', 'There', 'Then', 'They', 'These', 'Those', 'When', 'Where', 'Why', 'What', 'Who', 'How']:
                        sentence = current_sentence.strip()
                        if sentence: sentences.append(sentence)
                        current_sentence = ""
                        continue

            if char in '.!?':
                next_chars = text[i + 1:i + 4]
                quote_offset = 0
                if next_chars and next_chars[0] in '"\'':
                    quote_offset = 1
                    if len(next_chars) > 1 and next_chars[1] in '"\'': quote_offset = 2

                is_sentence_end = True
                remaining_text = text[i + 1 + quote_offset:].lstrip()
                if remaining_text:
                    next_char = remaining_text[0]
                    if next_char.islower():
                        whitespace_after = len(text[i + 1:]) - len(text[i + 1:].lstrip())
                        if whitespace_after < 2 and quote_offset == 0: is_sentence_end = False
                    if char == '.' and i > 0:
                        words_before = current_sentence.strip().split()
                        if words_before and len(words_before[-1]) <= 4 and words_before[-1].isupper(): is_sentence_end = False

                if is_sentence_end:
                    while i + 1 < len(text) and text[i + 1] in '"\'':
                        i += 1
                        current_sentence += text[i]
                    sentence = current_sentence.strip()
                    if sentence: sentences.append(sentence)
                    current_sentence = ""
            i += 1

        if current_sentence.strip():
            sentences.append(current_sentence.strip())

        for i, sentence in enumerate(sentences):
            for placeholder, original in protected_abbrevs.items():
                sentence = sentence.replace(placeholder, original)
            sentences[i] = sentence

        clean_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence: continue
            if (len(clean_sentences) > 0 and len(sentence.split()) <= 3 and
                not sentence[-1] in '.!?' and not sentence.startswith('"')):
                clean_sentences[-1] += " " + sentence
            else:
                clean_sentences.append(sentence)
        return clean_sentences

    def chunk_text(self, text: str) -> list[str]:
        """
        Packs WHOLE sentences into chunks.
        It will only break a sentence if that single sentence exceeds max_chars.
        """
        sentences = self._split_into_sentences(text)
        chunks = []
        current_chunk = ""

        for s in sentences:
            # If the chunk + this sentence is still within limits, combine them
            if len(current_chunk) + len(s) + 1 <= self.max_chars:
                current_chunk = (current_chunk + " " + s).strip()
            else:
                # If current_chunk has content, save it before moving to the next
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""

                # If this single sentence is still longer than max_chars, we must split it
                if len(s) > self.max_chars:
                    # Smart split at commas or spaces to minimize the 'cut' feel
                    sub_parts = re.split(r'(?<=,)\s+| ', s)
                    temp = ""
                    for part in sub_parts:
                        if len(temp) + len(part) + 1 > self.max_chars:
                            if temp: chunks.append(temp.strip())
                            temp = part
                        else:
                            temp = (temp + " " + part).strip()
                    current_chunk = temp
                else:
                    # Sentence fits in a new empty chunk
                    current_chunk = s

        if current_chunk:
            chunks.append(current_chunk)

        return chunks