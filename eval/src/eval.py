import re
from ml3m.base import BaseEvaluator


class McqRegexEvaluator(BaseEvaluator):

    _pats1 = [
        r"答案应?该?[为是]([ABCD,，、\s]+)",
        r"答案是([ABCD,，、\s]+)",
        r"选项应?该?[是为]([ABCD,，\s、]+)",
        r"答案([ABCD,，、\s]+)",
        r"答案应该?选\s?([ABCD,，、\s]+)",
        r"选择([ABCD,，、\s]+)",
        r"故?选[:：]?([ABCD,，、\s]+)",
        r"选择答案([ABCD,，\s、]+)",
        r"^\s*选([ABCD,，、\s]+)[,，项．\.]",
        r"([ABCD,，、\s]+)</s>",
        r"^([ABCD,，、\s]+)。",
    ]

    _pats2 = [
        r"([ABCD,，、\s]+)都?是正确的",
        r"选项([ABCD,，\s、]+)正确",
        r"([ABCD,，、\s]+)选项正确",
        r"([ABCD,，、\s]+)\s?都?[是为]正确答案",
        r"([ABCD,，、\s]+)\s?都?[是为]正确选项",
        r"([ABCD,，、\s]+)\s?选?项?都?属于[^错不]",
    ]

    def __init__(
        self,
        dataset,
        save_path,
        subjects,
        response_name,
        logging_mode="all",
        verbose=0,
    ):
        super().__init__(
            dataset=dataset,
            save_path=save_path,
            subjects=subjects,
            fmt="csv",
            workers=1,
            n_iter=1,
            agg_method=None,
            logging_mode=logging_mode,
            verbose=verbose,
        )
        self.response_name = response_name

    def _extract_answer(self, response):
        if len(response) < 10:
            return response

        for pat in self._pats1:
            mat = re.search(pat, response)
            if mat is not None:
                return mat.group(1).strip()

        mats = re.findall(r"([ABCD][\.．])\s?", response)
        if mats:
            return "".join(mats).strip()

        for pat in self._pats2:
            mats = re.findall(pat, response)
            if mats:
                return "".join(mats).strip()
        return response

    def _get_score(self, data_item, **kwargs):
        gt, response = data_item[["output", self.response_name]]
        answer = self._extract_answer(response)
        answer_set = set()

        for char in answer:
            if char in "ABCD":
                answer_set.add(char)

        return (set(gt) == answer_set) * 100
