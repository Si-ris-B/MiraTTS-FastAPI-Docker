import re
import logging

logger = logging.getLogger(__name__)


class TextProcessor:
    def __init__(self, max_chars: int = 250):
        self.max_chars = max_chars

    def split_into_sentences(self, text: str):
        if not text or not text.strip(): return []
        text = re.sub(r'\s+', ' ', text.strip())

        # Robust abbreviations protection
        abbreviations = {'Mr.', 'Mrs.', 'Ms.', 'Dr.', 'Prof.', 'Sr.', 'Jr.', 'vs.', 'etc.', 'U.S.', 'U.K.'}
        protected = {}
        for i, abbrev in enumerate(abbreviations):
            placeholder = f"__ABBR_{i}__"
            if abbrev in text:
                text = text.replace(abbrev, placeholder)
                protected[placeholder] = abbrev

        sentences = re.split(r'(?<=[.!?])\s+', text)

        final = []
        for s in sentences:
            for p, orig in protected.items():
                s = s.replace(p, orig)
            if s.strip(): final.append(s.strip())
        return final

    def chunk_text(self, text: str) -> list[str]:
        sentences = self.split_into_sentences(text)
        chunks = []
        current = ""
        for s in sentences:
            if len(s) > self.max_chars:
                if current: chunks.append(current.strip()); current = ""
                words = s.split(' ')
                for w in words:
                    if len(current) + len(w) + 1 > self.max_chars:
                        chunks.append(current.strip());
                        current = w
                    else:
                        current = f"{current} {w}".strip()
            elif len(current) + len(s) + 1 > self.max_chars:
                chunks.append(current.strip());
                current = s
            else:
                current = f"{current} {s}".strip()
        if current: chunks.append(current.strip())
        return chunks