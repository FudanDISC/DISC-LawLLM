import asyncio
import json
import os
import re

import openai
import pandas as pd
import torch
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaForCausalLM,
    LlamaTokenizer,
)
from transformers.generation.utils import GenerationConfig


# Try to read few shot examples, otherwise do not support mcq
dirname = os.path.dirname(__file__)
support_mcq_sing, support_mcq_mult = True, True

try:
    mcq_sing_path = os.path.join(dirname, "few_shot", "mcq_sing_examples.csv")
    df_sing_few_shot = pd.read_csv(mcq_sing_path)
except Exception as e:
    support_mcq_sing = False
    support_mcq_sing_e = e

try:
    mcq_mult_path = os.path.join(dirname, "few_shot", "mcq_mult_examples.csv")
    df_mult_few_shot = pd.read_csv(mcq_mult_path)
except Exception as e:
    support_mcq_mult = False
    support_mcq_mult_e = e


class BaseModel:
    """The base model class."""

    def __init__(self):
        pass

    def chat(self, query):
        """Return the response to the query.

        This function is intended to be overridden in subclasses.

        Parameters
        ----------
        query : str
            The query.

        Returns
        -------
        response : str
            The response to the query.
        """
        raise NotImplementedError

    async def achat(self, query):
        """Return the response to the query.

        This function should be asynchronous, and is intended to be overridden in
        subclasses.

        Parameters
        ----------
        query : str
            The query.

        Returns
        -------
        response : str
            The response to the query.
        """
        raise NotImplementedError

    def mcq_formatter(self, multi, n_shot):
        """Format query for multiple-choice questions.

        Parameters
        ----------
        multi : bool
            Whether the multiple-choice question has multiple correct options.

        Returns
        -------
        formatter : str
            The string to format the multiple-choice question query. It should contain
            "{question}" and "{options}" used for formatting.
        """
        if multi:
            if not support_mcq_mult:
                raise support_mcq_mult_e
            kind, examples_df = "多项", df_mult_few_shot
        else:
            if not support_mcq_sing:
                raise support_mcq_sing_e
            kind, examples_df = "单项", df_sing_few_shot

        # Check that we are not requiring more few shot examples than supported
        if n_shot > len(examples_df):
            raise ValueError(
                f"MCQ ({kind}) supports at most {len(examples_df)} few shot examples; "
                f"got {n_shot}"
            )

        contents = [
            f"以下是一道{kind}选择题，不需要做任何分析和解释，直接输出答案选项。\n"
            f"下面给出了{n_shot}个样例，按照此样例输出答案。"
        ]
        for _, row in examples_df.head(n_shot).iterrows():
            example_q, example_a = row[["input", "output"]]
            contents.append(f"问题：{example_q}\n答案：{example_a}")
        contents.append("下面是问题：\n{question}\n{options}\n答案：")
        return "\n\n".join(contents)




    def qa_formatter(self):
        """Format query for question-answering.

        Returns
        -------
        formatter : str
            The string to format the question-answering query. It should contain
            "{question}" used for formatting.
        """
        return "{question}"


class GPTModel(BaseModel):
    """
    gpt-4 : GPT-4
        https://platform.openai.com/docs/models/gpt-4
    gpt-3.5-turbo: GPT-3.5 Turbo
        https://platform.openai.com/docs/models/gpt-3-5
    """

    def __init__(self, ver):
        self.ver = ver
        if self.ver not in ["gpt-4", "gpt-3.5-turbo"]:
            raise ValueError(f"Invalid 'ver'; got '{self.ver}'")

        # Configure OpenAI API
        dirname = os.path.dirname(__file__)
        try:
            api_path = os.path.join(dirname, "..", "openai.json")
            with open(api_path, "r", encoding="utf-8") as f:
                apis = json.load(f)[0]
            openai.api_key = apis["key"]
            if apis["base"] is not None:
                openai.api_base = apis["base"]
        except FileNotFoundError:
            raise FileNotFoundError(
                f"{type(self).__name__} requires an OpenAI configuration file at "
                f"{os.path.abspath(api_path)}. See "
                "https://charlie-xiao.github.io/ml3m/modules/ml3m.utils.openai.html#examples"
                "for an example, and the first API will be used."
            )

    def _format_message(self, query):
        return [
            {"role": "system", "content": "你是一名中国法律专家。"},
            {"role": "user", "content": query},
        ]

    def _process_completion(self, completion):
        stop_reason = completion["choices"][0]["finish_reason"]
        if stop_reason != "stop":
            raise ValueError(f"Model terminated due to '{stop_reason}'")
        return completion["choices"][0]["message"]["content"]

    def chat(self, query):
        completion = openai.ChatCompletion.create(
            model=self.ver,
            messages=self._format_message(query),
        )
        return self._process_completion(completion)

    async def achat(self, query):
        completion = await asyncio.wait_for(
            openai.ChatCompletion.acreate(
                model=self.ver,
                messages=self._format_message(query),
            ),
            timeout=60,
        )
        return self._process_completion(completion)


class LaWGPTModel(BaseModel):
    """
    https://github.com/pengxiao-song/LaWGPT
    """

    def __init__(self):
        # TODO(zyx): model initialization code
        self.model = ...
        self.tokenizer = ...

    def chat(self, query):
        # TODO(zyx): query (str) -> response (str)
        return ...

    def mcq_formatter(self, multi):
        # TODO(zyx): format query for mcq; modify this if you need something different
        # from the default implementation for this model; otherwise delete this
        return ...

    def qa_formatter(self):
        # TODO(zyx): format query for mcq; modify this if you need something different
        # from the default implementation for this model; otherwise delete this
        return ...


class LawyerLLaMAModel(BaseModel):
    """
    https://github.com/AndrewZhe/lawyer-llama
    """

    def __init__(self):
        # TODO(zyx): model initialization code
        self.model = ...
        self.tokenizer = ...

    def chat(self, query):
        # TODO(zyx): query (str) -> response (str)
        return ...

    def mcq_formatter(self, multi):
        # TODO(zyx): format query for mcq; modify this if you need something different
        # from the default implementation for this model; otherwise delete this
        return ...

    def qa_formatter(self):
        # TODO(zyx): format query for mcq; modify this if you need something different
        # from the default implementation for this model; otherwise delete this
        return ...


class LexiLawModel(BaseModel):
    """
    https://github.com/CSHaitao/LexiLaw
    """

    def __init__(self):
        # TODO(zyx): model initialization code
        self.model = ...
        self.tokenizer = ...

    def chat(self, query):
        # TODO(zyx): query (str) -> response (str)
        return ...

    def mcq_formatter(self, multi):
        # TODO(zyx): format query for mcq; modify this if you need something different
        # from the default implementation for this model; otherwise delete this
        return ...

    def qa_formatter(self):
        # TODO(zyx): format query for mcq; modify this if you need something different
        # from the default implementation for this model; otherwise delete this
        return ...


class ChineseAlpacaModel(BaseModel):
    """
    pro-7b : Chinese-Alpaca-Pro-7B
        https://github.com/ymcui/Chinese-LLaMA-Alpaca
    2-7b: Chinese-Alpaca-2-7B
        https://github.com/ymcui/Chinese-LLaMA-Alpaca-2
    2-13b: Chinese-Alpaca-2-13B
        https://github.com/ymcui/Chinese-LLaMA-Alpaca-2
    """

    def __init__(self, ver):
        self.ver = ver
        if self.ver == "pro-7b":
            mpath = ""  # Use your path for Chinese-Alpaca-Pro-7B
            self.tokenizer = AutoTokenizer.from_pretrained(mpath)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.unk_token
        elif self.ver == "2-7b":
            mpath = ""  # Use your path for Chinese-Alpaca-2-7B
            self.tokenizer = AutoTokenizer.from_pretrained(mpath)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.unk_token
        elif self.ver == "2-13b":
            mpath = ""  # Use your path for Chinese-Alpaca-2-13B
            self.tokenizer = LlamaTokenizer.from_pretrained(mpath, legacy=True)
        else:
            raise ValueError(f"Invalid 'ver'; got '{self.ver}'")

        self.model = LlamaForCausalLM.from_pretrained(
            mpath,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.generation_config = GenerationConfig(
            temperature=0.1,
            top_k=5,
            top_p=0.85,
            repetition_penalty=1.1,
            max_new_tokens=1024,
        )

    def chat(self, query):
        query = (
            "[INST] <<SYS>>\nYou are a helpful assistant. 你是一个乐于助人的助手。\n"
            f"<</SYS>>\n\n{query.strip()} [/INST]"
        )
        inputs = self.tokenizer(query, return_tensors="pt")
        response = self.model.generate(
            inputs.input_ids.to(self.model.device),
            self.generation_config,
            attention_mask=inputs["attention_mask"].to(self.model.device),
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        output = self.tokenizer.decode(response[0], skip_special_tokens=True)
        return output[len(query) :].strip("</s>").strip()


class BaichuanModel(BaseModel):
    """
    7b : Baichuan-7B
        https://github.com/baichuan-inc/baichuan-7B
    13b-base : Baichuan-13B-Base
        https://github.com/baichuan-inc/Baichuan-13B
    13b-chat : Baichuan-13B-Chat
        https://github.com/baichuan-inc/Baichuan-13B
    2-13b-chat : Baichuan2-13B-Chat
        https://github.com/baichuan-inc/Baichuan2
    """

    def __init__(self, ver):
        self.ver = ver
        if self.ver == "7b":
            mpath = ""  # Use your path for Baichuan-7B
            self.tokenizer = AutoTokenizer.from_pretrained(
                mpath, trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                mpath,
                load_in_8bit=False,
                low_cpu_mem_usage=True,
                device_map="auto",
                trust_remote_code=True,
            )
            self.generation_config = {
                "temperature": 0.1,
                "top_k": 5,
                "top_p": 0.85,
                "repetition_penalty": 1.1,
                "max_new_tokens": 1024,
                "pad_token_id": 0,
                "bos_token_id": 1,
                "eos_token_id": 2,
            }
        elif self.ver == "13b-base":
            mpath = ""  # Use your path for Baichuan-13B-Base
            self.tokenizer = AutoTokenizer.from_pretrained(
                mpath, trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                mpath,
                load_in_8bit=False,
                low_cpu_mem_usage=True,
                device_map="auto",
                trust_remote_code=True,
            )
            self.model.generation_config = GenerationConfig.from_pretrained(mpath)
            self.model.generation_config.user_token_id = 195
            self.model.generation_config.assistant_token_id = 196
            self.model.generation_config.pad_token_id = (
                self.model.generation_config.eos_token_id
            )
        elif self.ver == "13b-chat":
            mpath = ""  # Use your path for Baichuan-13B-Chat
            self.tokenizer = AutoTokenizer.from_pretrained(
                mpath, trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                mpath,
                load_in_8bit=False,
                low_cpu_mem_usage=True,
                device_map="auto",
                trust_remote_code=True,
            )
            self.model.generation_config = GenerationConfig.from_pretrained(mpath)
        elif self.ver == "2-13b-chat":
            mpath = ""  # Use your path for Baichuan2-13B-Chat
            self.tokenizer = AutoTokenizer.from_pretrained(
                mpath, trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                mpath,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
            self.model.generation_config = GenerationConfig.from_pretrained(mpath)
        else:
            raise ValueError(f"Invalid 'ver'; got '{self.ver}'")

    def chat(self, query):
        if self.ver in ["7b"]:
            inputs_id = self.tokenizer.encode(query, return_tensors="pt")
            with torch.no_grad():
                generation_output = self.model.generate(
                    inputs_id.to(self.model.device), **self.generation_config
                )
            return self.tokenizer.decode(
                generation_output[0, inputs_id.shape[-1] :], skip_special_tokens=True
            )
        return self.model.chat(self.tokenizer, [{"role": "user", "content": query}])


class ChatGLMModel(BaseModel):
    """
    6b: ChatGLM-6B
        https://github.com/THUDM/ChatGLM-6B
    2-6b: ChatGLM2-6B
        https://github.com/THUDM/ChatGLM2-6B
    """

    def __init__(self, ver):
        self.ver = ver
        if self.ver == "6b":
            mpath = ""  # Use your path for ChatGLM-6B
            self.model = AutoModel.from_pretrained(
                mpath, trust_remote_code=True, device="cuda"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                mpath, trust_remote_code=True
            )
            self.generation_config = {
                "temperature": 0.3,
                "top_k": 5,
                "top_p": 0.85,
                "repetition_penalty": 1.1,
                "max_new_tokens": 1024,
            }
        elif self.ver == "2-6b":
            mpath = ""  # Use your path for ChatGLM2-6B
            self.model = (
                AutoModel.from_pretrained(
                    mpath,
                    trust_remote_code=True,
                    device_map="auto",
                )
                .half()
                .eval()
            )  # Note that it is halved before eval
            self.tokenizer = AutoTokenizer.from_pretrained(
                mpath, trust_remote_code=True
            )
            self.generation_config = {
                "temperature": 0.1,
                "top_k": 5,
                "top_p": 0.85,
                "repetition_penalty": 1.1,
                "max_new_tokens": 1024,
            }
        else:
            raise ValueError(f"Invalid 'ver'; got '{self.ver}'")

    def chat(self, query):
        response, _ = self.model.chat(
            self.tokenizer, query.strip(), history=[], **self.generation_config
        )
        return response


class ChatLawModel(BaseModel):
    """
    https://github.com/PKU-YuanGroup/ChatLaw
    """

    def __init__(self):
        try:
            from peft import PeftModel
        except ImportError:
            raise ImportError("The peft package is required for ChatLawModel")

        mpath = ""  # Use your path for ChatLaw
        base_mpath = ""  # Use your base model path for ChatLaw
        model = LlamaForCausalLM.from_pretrained(
            base_mpath,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.model = PeftModel.from_pretrained(model, mpath)
        self.generation_config = GenerationConfig(
            temperature=0.3,
            top_k=5,
            top_p=0.85,
            repetition_penalty=1.1,
            max_new_tokens=1024,
        )
        self.tokenizer = LlamaTokenizer.from_pretrained(mpath)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.unk_token

    def chat(self, query):
        prompt = f"Consult:\n{query.strip()}\nResponse:\n"
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        inputs["input_ids"] = inputs["input_ids"].to(self.model.device)
        with torch.no_grad():
            generation_output = self.model.generate(
                **inputs,
                generation_config=self.generation_config,
                return_dict_in_generate=True,
                output_scores=True,
            )
        output = self.tokenizer.decode(generation_output.sequences[0])

        mat = re.search("Response\s*:\s*([\s\S]+?)</s>", output)
        if mat is not None:
            return mat.group(1)
        mat = re.search("Response\s*:\s*(.*)$", output, re.S)
        if mat is not None:
            return mat.group(1)
        return output


class ChineseLLaMAModel(BaseModel):
    """
    2-13b : Chinese-LLaMA-2-13B
        https://github.com/ymcui/Chinese-LLaMA-Alpaca-2
    """

    def __init__(self, ver):
        self.ver = ver
        if self.ver == "2-13b":
            mpath = ""  # Use your path for Chinese-LLaMA-2-13B
            self.model = AutoModelForCausalLM.from_pretrained(
                mpath,
                trust_remote_code=True,
                device_map="auto",
                torch_dtype=torch.float16,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(mpath, use_fast=False)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.generation_config = GenerationConfig(
                temperature=0.3,
                top_k=5,
                top_p=0.85,
                repetition_penalty=1.1,
                max_new_tokens=1024,
            )
        else:
            raise ValueError(f"Invalid 'ver'; got '{self.ver}'")

    def chat(self, query):
        prompt = f"{query.strip()}\n答："
        input_ids = self.tokenizer(
            [prompt], return_tensors="pt", add_special_tokens=False
        ).input_ids.to("cuda")
        generate_ids = self.model.generate(input_ids, self.generation_config)
        return self.tokenizer.decode(generate_ids[0])[len(prompt) :].replace("</s>", "")


class DISCLawLLMModel(BaseModel):
    """
    13b : DISC-LawLLM-13B
        https://github.com/FudanDISC/DISC-LawLLM
    """

    def __init__(self, ver):
        self.ver = ver
        if self.ver == "13b":
            mpath = ""  # Use your path for DISC-LawLLM-13B
            self.tokenizer = AutoTokenizer.from_pretrained(
                mpath, trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                mpath,
                load_in_8bit=False,
                low_cpu_mem_usage=True,
                device_map="auto",
                trust_remote_code=True,
            ).eval()
            self.model.generation_config = GenerationConfig.from_pretrained(mpath)
            self.model.generation_config.user_token_id = 195
            self.model.generation_config.assistant_token_id = 196
            self.tokenizer.user_token_id = 195
            self.tokenizer.assistant_token_id = 196
        else:
            raise ValueError(f"Invalid 'ver'; got '{self.ver}'")

    def chat(self, query):
        return self.model.chat(self.tokenizer, [{"role": "user", "content": query}])


class ZhiHaiLuWenModel(BaseModel):
    """
    https://github.com/zhihaiLLM/wisdomInterrogatory
    """

    def __init__(self):
        mpath = ""  # Use your path for Zhi Hai Lu Wen
        self.tokenizer = AutoTokenizer.from_pretrained(mpath, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            mpath, device_map="auto", trust_remote_code=True
        ).half()  # Note that it is halved

    def chat(self, query):
        torch.cuda.empty_cache()
        inputs = self.tokenizer(
            f"</s>Human:{query} </s>Assistant: ", return_tensors="pt"
        )
        inputs = inputs.to("cuda")
        with torch.no_grad():
            pred = self.model.generate(
                **inputs, max_new_tokens=800, repetition_penalty=1.2
            )
        response = self.tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)
        return response.split("Assistant: ")[1]


# Models included in the current evaluation

MODELS = {
    "chatglm": {
        "klass": ChatGLMModel,
        "kwargs": {"ver": "6b"},
    },
    "baichuan-chat": {
        "klass": BaichuanModel,
        "kwargs": {"ver": "13b-chat"},
    },
    "chinese-alpaca-2": {
        "klass": ChineseAlpacaModel,
        "kwargs": {"ver": "2-13b"},
    },
    "gpt-3.5-turbo": {
        "klass": GPTModel,
        "kwargs": {"ver": "gpt-3.5-turbo"},
    },
    "lexilaw": {
        "klass": BaseModel,  # TODO
    },
    "lawgpt": {
        "klass": BaseModel,  # TODO
    },
    "lawyerllama": {
        "klass": BaseModel,  # TODO
    },
    "chatlaw": {
        "klass": ChatLawModel,
    },
    "disclawllm": {
        "klass": DISCLawLLMModel,
        "kwargs": {"ver": "13b"},
    },
}

# Models that are not included in the current evaluation. They are either different
# versions of already-evaluated models but underperform their variants, or are models
# whose outputs cannot be properly obtained due to technical issues

_OTHER_MODELS = {
    "-chatglm2-6b": {
        "klass": ChatGLMModel,
        "kwargs": {"ver": "2-6b"},
    },
    "-baichuan-chat-7b": {
        "klass": BaichuanModel,
        "kwargs": {"ver": "7b"},
    },
    "-baichuan-chat-13b-base": {
        "klass": BaichuanModel,
        "kwargs": {"ver": "13b-base"},
    },
    "-baichuan2-13b-chat": {
        "klass": BaichuanModel,
        "kwargs": {"ver": "2-13b-chat"},
    },
    "-chinese-alpaca-pro-7b": {
        "klass": ChineseAlpacaModel,
        "kwargs": {"ver": "pro-7b"},
    },
    "-chinese-alpaca-2-7b": {
        "klass": ChineseAlpacaModel,
        "kwargs": {"ver": "2-7b"},
    },
    "-gpt-4": {
        "klass": GPTModel,
        "kwargs": {"ver": "gpt-4"},
    },
    "-zhihailuwen": {
        "klass": ZhiHaiLuWenModel,
    },
    "-chinese-llama-2-13b": {
        "klass": ChineseLLaMAModel,
        "kwargs": {"ver": "2-13b"},
    },
}

MODELS.update(_OTHER_MODELS)


def get_model(model_name):
    """Get a model instance by the model name.

    Parameters
    ----------
    model_name : str
        The name of the model to get.

    Returns
    -------
    model : subclass of BaseModel
        The model instance.
    """
    model = MODELS[model_name]
    if "kwargs" in model:
        return model["klass"](**model["kwargs"])
    return model["klass"]()
