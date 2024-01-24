import asyncio
import json
import openai
import os
import re
from numbers import Real
from typing import Callable, Literal, Any

import pandas as pd
from ml3m.base import BaseEvaluator, BaseOpenAIEvaluator
from openai import AsyncOpenAI
from pathlib import Path


AggregateMethod = Literal["mean", "sum", "min", "max", "mode"]
DataItemType = pd.Series | list | dict
DatasetFormat = Literal["jsonl", "json", "csv"]
LoggingMode = Literal["all", "failed", "none"]


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


class QaOpenaiEvaluator(BaseOpenAIEvaluator):
    """Evaluator for question-answering via OpenAI.

    This evaluator utilizes the ability of OpenAI models to tell the quality of a
    response from the following aspects:

    - **Accuracy**: Using the reference answer as the ground truth, does the response
      include factually incorrect information?
    - **Completeness**: Compared with the reference answer, is the response missing
      details?
    - **Clarity**: Is the response well-organized and clearly presented? If accuracy
      and completeness is poor, clarity should also be considered poor.

    Parameters
    ----------
    dataset : str or pathlib.Path
        The absolute path to the evaluation dataset.
    save_path : str or pathlib.Path
        The absolute path to the save location. This path may or may not exist, and if
        it exists, its file contents will be treated as a (partially) written result.
        Whether to overwrite the existing results or to build on them depend on
        ``overwrite`` when using the :meth:`QaOpenAIEvaluator.evaluate` method.
    openai_config : str or pathlib.Path
        The absolute path to the OpenAI configuration file.
    info_func : Callable
        The function that extracts the question, actual answer, and expected answer of
        a data item. The input parameter should be a :class:`pandas.Series`, a list, or
        a dictionary, depending on ``fmt`` and the specific type of each data item. The
        output should be a tuple of three strings, respectively the question, the actual
        answer to that question, and the expected answer of that question. See the notes
        for examples.
    fmt : {"jsonl", "json", "csv"}, default="jsonl"
        The format of ``dataset``.
    domain : str, optional
        The domain of knowledge. ChatGPT will be prompted to know that your question,
        answer, and reference answer are "in {domain}". If ``None``, then this
        information will not be given to ChatGPT.
    aspects : list of str, optional
        The aspects to evaluate. If ``None``, evalute accuracy, completeness, and
        clarity. If there is any string other than "accuracy", "completeness", and
        "clarity", then they have to be specified in ``aspect_descriptions``.
    aspect_descriptions : dict, optional
        An optional dictionary mapping aspects to their descriptions. "accuracy",
        "completeness", and "clarity" have default descriptions but can also be
        overridden by this parameter. Any other aspect, if used in ``aspects``, must
        exist as a key here.
    n_iter : int, default=3
        The number of iterations for each data item. The mean of the scores for each
        data item will be taken as the final score.
    timeout : float, default=60
        The timeout in seconds. This is not the OpenAI timeout, but the timeout for
        cancelling the worker tasks.
    model : str, default="gpt-3.5-turbo"
        The ID of the model to use, must be one of the available OpenAI models that
        support the ChatCompletion API. See also
        https://platform.openai.com/docs/models/model-endpoint-compatibility
    logging_mode : {"all", "failed", "none"}, default="all"
        The logging mode, whether to save the logs of all items, or only of failed
        items, or save no log.
    verbose : int, default=0
        The verbosity level of the processing. For negative levels, only a progress bar
        will be displayed. For level 0, the errored items will also be displayed. For
        positive levels, the all items will be displayed, and the verbosity level
        determines the number of lines to display for the message of each item.

    Notes
    -----
    Here are some examples of ``info_func``:

    Assume that ``dataset`` is in ``.jsonl`` format and each line is of the following
    form: ``{{"instruction": "xxx", "input": "xxx", "output": "xxx", "history": [],
    "response": "xxx"}}``. Then ``info_func`` can be defined as follows:

    .. code-block:: python

        def info_func(data_item: dict) -> tuple[str, str, str]:
            question = data_item["instruction"] + "\\n" + data_item["input"]
            actual = data_item["response"]
            expected = data_item["output"]
            return question, actual, expected

    Now assume that ``dataset`` is in ``.csv`` format with columns "question",
    "answer", and "response". Then ``info_func`` can be defined as follows:

    .. code-block:: python

        def info_func(data_item: pandas.Series) -> tuple[str, str, str]:
            question, answer, response = data_item[["question", "answer", "response"]]
            return question, response, answer
    """

    _pattern = re.compile(r"```[a-z]*\n(.+)\n```")

    def __init__(
        self,
        dataset: str | Path,
        save_path: str | Path,
        openai_config: str | Path,
        info_func: Callable[DataItemType, tuple[str, str, str]],
        *,
        fmt: DatasetFormat = "jsonl",
        domain: str | None = None,
        aspects: list[str] | None = None,
        aspect_descriptions: dict[str, str] | None = None,
        n_iter: int = 3,
        timeout: float = 60,
        model: str = "gpt-3.5-turbo-0613",
        logging_mode: LoggingMode = "all",
        verbose: int = 0,
    ) -> None:
        self.info_func = info_func
        self.domain = domain

        # Determine the aspects to evaluate on
        avail_aspects: list[str] = ["accuracy", "completeness", "clarity"]
        self.aspects = avail_aspects if aspects is None else aspects
        self.aspect_descriptions = {
            "accuracy": (
                "The content of pending scored answer conforms to reference answer in semantic,"
                "especially noting the content of the law, the facts in question and the conclusion."
            ),
            "completeness": (
                "Compared to reference answer, pending scored answer does not miss any details in reference answer.Do not let the length of the answer influence your judgment."
            ),
            "clarity": (
                "the logic of pending scored answer is rigorous and clear, and the sentences are well-organized. "
                "If Accuracy and Completeness is bad, Clarity should also be bad."
            ),
        }
        if aspect_descriptions is not None:
            self.aspect_descriptions.update(aspect_descriptions)

        # Validate the arguments
        if not callable(self.info_func):
            raise InvalidParameterError(
                "info_func", actual=self.info_func, reason="must be a callable"
            )
        if not isinstance(self.aspects, list):
            raise InvalidParameterError(
                "aspects", actual=self.aspects, reason="must be a list"
            )
        if (
            any(subject not in self.aspect_descriptions for subject in self.aspects)
            or len(self.aspects) != len(set(self.aspects))
            or len(self.aspects) == 0
        ):
            raise InvalidParameterError(
                "aspects",
                actual=self.aspects,
                reason=(
                    "must be a list of non-duplicated aspects among "
                    f"{list(self.aspect_descriptions)}"
                ),
            )

        # Set the subject explanations
        self._explanations = [
            f"{subject}: {self.aspect_descriptions[subject]}"
            for subject in self.aspects
        ]
        self._domain = "" if self.domain is None else f" in {self.domain}"

        # Inherit from parent
        super().__init__(
            dataset=dataset,
            save_path=save_path,
            subjects=self.aspects,
            openai_config=openai_config,
            fmt=fmt,
            n_iter=n_iter,
            agg_method="mean",
            timeout=timeout,
            model=model,
            logging_mode=logging_mode,
            verbose=verbose,
        )
        with open(openai_config, "r", encoding="utf-8") as f:
            apis = json.load(f)[0]
        # openai.api_key = apis["key"]
        os.environ["OPENAI_API_KEY"] = apis["key"]
        if apis["base"] is not None:
            # openai.api_base = apis["base"]
            os.environ["OPENAI_API_BASE"] = apis["base"]
        self.client = AsyncOpenAI(api_key=apis["key"], base_url=apis["base"])

    def _prompt(self, data_item: DataItemType) -> tuple[str, str]:
        """:meta private:"""
        question, actual, expected = self.info_func(data_item)
        explanation_expr = "\n".join(
            [
                f"{i + 1}. {explanation}"
                for i, explanation in enumerate(self._explanations)
            ]
        )
        return (
            "You are a professional, impartial, and strict scorer. You will be given "
            "a question, a pending scored answer, and a reference answer "
            f"{self._domain}. Please rate the pending scored answer based on the "
            f"reference answer in the following aspects:\n{explanation_expr}\n\nEach "
            "score should be from 1 (lowest)-5 (highest). Your rating should be strict enough, and do "
            "not easily give full scores. Pay attention to the difference between laws mentioned in the reference answer and the pending scored answer. In your response, you should only include a "
            "JSON object, with keys being the aspects and values being the scores. Do "
            "not include any additional information or explanation.",
            f"### Question\n```\n{question}\n```\n\n### Reference answer\n```\n"
            f"{expected}\n```\n\n### Pending scored answer\n```\n{actual}\n```",
        )

    def _extract_scores(
        self, reply: str, data_item: DataItemType
    ) -> Real | dict[Any, Real]:
        """:meta private:"""
        scores: dict[str, Real]
        try:
            # reply = reply[7:-3].strip()
            scores = {
                subject.lower(): score for subject, score in json.loads(reply).items()
            }

        # Try to search for a code block in the reply; if errored, leave as is
        except:
            match = re.search(self._pattern, reply)
            assert match is not None, "backup pattern matching failed."
            scores = {
                subject.lower(): score
                for subject, score in json.loads(match.group(1)).items()
            }
        return scores

    def _post_scoring(
        self, completion: Any, data_item: DataItemType
    ) -> Real | dict[Any, Real]:
        """Process the OpenAI response posterior to querying.

        Parameters
        ----------
        completion : an OpenAI completion
            The returned OpenAI completion.

        Returns
        -------
        scores : real or dict
            The extracted scores, either a single score or a dictionary of subject-
            score pairs.
        """
        return self._extract_scores(
            completion.choices[0].message.content, data_item
        )

    async def _aget_score(
        self, data_item: DataItemType, **kwargs
    ) -> Real | dict[Any, Real]:
        """:meta private:"""
        messages = self._prior_scoring(data_item)
        completion = await asyncio.wait_for(
            self.client.chat.completions.create(
                model=self.model,
                messages=messages,
            ),
            timeout=self.timeout,
        )
        return self._post_scoring(completion, data_item)
