import re
import sys
import unicodedata
from lmms_eval.api.filter import Filter

class WhitespaceFilter(Filter):

    def __init__(self) -> None:
        pass

    def apply(self, resps, docs):

        def filter_set(inst):
            filtered_resp = []
            for resp in inst:
                if resp.startswith(' '):
                    resp = resp[1:]
                filtered_resp.append(resp)
            return filtered_resp
        filtered_resps = [filter_set(resp) for resp in resps]
        return filtered_resps

class RegexFilter(Filter):

    def __init__(self, regex_pattern: str='#### (\\-?[0-9\\.\\,]+)', group_select=0, fallback: str='[invalid]') -> None:
        self.regex_pattern = regex_pattern
        self.regex = re.compile(regex_pattern)
        self.group_select = group_select
        self.fallback = fallback

    def apply(self, resps, docs):

        def filter_set(inst):
            filtered = []
            for resp in inst:
                match = self.regex.findall(resp)
                if match:
                    match = match[self.group_select]
                    if isinstance(match, tuple):
                        match = [m for m in match if m][0]
                    match = match.strip()
                else:
                    match = self.fallback
                filtered.append(match)
            return filtered
        filtered_resps = list(map(lambda x: filter_set(x), resps))
        return filtered_resps

class MultiChoiceRegexFilter(RegexFilter):

    def __init__(self, regex_pattern: str='#### (\\-?[0-9\\.\\,]+)', group_select=0, fallback: str='[invalid]', ignore_case=False, ignore_punctuation=False, regexes_to_ignore=None) -> None:
        super().__init__(regex_pattern, group_select, fallback)
        self.ignore_case = ignore_case
        self.ignore_punctuation = ignore_punctuation
        self.regexes_to_ignore = regexes_to_ignore

    def apply(self, resps, docs):

        def find_match(regex, resp, convert_dict={}):
            match = regex.findall(resp)
            if match:
                match = match[self.group_select]
                if isinstance(match, tuple):
                    match = [m for m in match if m][0]
                match = match.strip()
                if match and match in convert_dict:
                    match = convert_dict[match]
            return match
        punct_tbl = dict.fromkeys((i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P')))

        def filter_ignores(st):
            if self.regexes_to_ignore is not None:
                for s in self.regexes_to_ignore:
                    st = re.sub(s, '', st)
            if self.ignore_case:
                st = st.lower()
            if self.ignore_punctuation:
                st = st.translate(punct_tbl)
            return st
        filtered_resps = []
        for r, doc in zip(resps, docs):
            fallback_regexes = []
            choice_to_alpha = {}
            next_alpha = 'A'
            without_paren_fallback_regexes = []
            without_paren_to_target = {}
            choices = doc['choices']
            for c in choices:
                m = filter_ignores(c.strip())
                fallback_regexes.append(f'{re.escape(m)}')
                choice_to_alpha[m] = f'({next_alpha})'
                without_paren_fallback_regexes.append(next_alpha)
                without_paren_to_target[next_alpha] = f'({next_alpha})'
                next_alpha = chr(ord(next_alpha) + 1)
            fallback_regex = re.compile('|'.join(fallback_regexes))
            without_paren_fallback_regex = '|'.join(without_paren_fallback_regexes)
            without_paren_fallback_regex = re.compile(f':[\\s]*({without_paren_fallback_regex})')
            filtered = []
            for resp in r:
                match = find_match(self.regex, resp)
                if not match:
                    match = find_match(fallback_regex, filter_ignores(resp), choice_to_alpha)
                    if not match:
                        match = find_match(without_paren_fallback_regex, resp, without_paren_to_target)
                if not match:
                    match = self.fallback
                filtered.append(match)
            filtered_resps.append(filtered)
        return filtered_resps

class ExtendedRegexFilter(RegexFilter):
    punct_tbl = dict.fromkeys((i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P')))

    def __init__(self, regex_pattern: str='#### (\\-?[0-9\\.\\,]+)', group_select=0, fallback: str='[invalid]', ignore_case=False, ignore_punctuation=False, regexes_to_ignore=None) -> None:
        super().__init__(regex_pattern, group_select, fallback)
        self.ignore_case = ignore_case
        self.ignore_punctuation = ignore_punctuation
        self.regexes_to_ignore = regexes_to_ignore

    def filter_ignores(self, st):
        if self.regexes_to_ignore is not None:
            for s in self.regexes_to_ignore:
                st = re.sub(s, '', st)
        if self.ignore_case:
            st = st.lower()
        if self.ignore_punctuation:
            st = st.translate(self.punct_tbl)
        return st

    def find_match(self, regex, resp, convert_dict={}):
        match = regex.findall(resp)
        if match:
            match = match[self.group_select]
            if isinstance(match, tuple):
                match = [m for m in match if m][0]
            match = match.strip()
            if match and match in convert_dict:
                match = convert_dict[match]
        return match

class SimpleMultiChoiceRegexFilter(ExtendedRegexFilter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def apply(self, resps, docs):
        filtered_resps = []
        for r, doc in zip(resps, docs):
            fallback_regexes = []
            choice_to_alpha = {}
            next_alpha = 'A'
            without_paren_fallback_regexes = []
            without_paren_to_target = {}
            multiple_choices_regex = re.compile('\\b([A-Z])\\.\\s+([^\\n]*)')
            matches = multiple_choices_regex.findall(doc['question'])
            for m in matches:
                choice_text = m[1].strip()
                fallback_regexes.append(f'{re.escape(choice_text)}')
                choice_to_alpha[choice_text] = next_alpha
                next_alpha = chr(ord(next_alpha) + 1)
            fallback_regex = re.compile('|'.join(fallback_regexes))
            filtered = []
            for resp in r:
                cleaned_resp = re.sub('[^\\w\\s]', '', resp).strip()
                match = fallback_regex.search(cleaned_resp)
                if match and match.group() in choice_to_alpha:
                    filtered.append(choice_to_alpha[match.group()])
                else:
                    filtered.append(cleaned_resp)
            filtered_resps.append(filtered[0])
        return filtered_resps
