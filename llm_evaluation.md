
[TOC]

## LLM Evaluation Reference

Traditionally, language model performance is measured by measuring **perplexity, cross entropy, and bits-per-character (BPC)**. Large Language Models, on the other hand, have been shown to outperform these benchmarks and unlock new abilities such as arithmetic, few-shot learning, and multi-step reasoning. Nevertheless, these LLMs are not without flaws, exhibiting biases and producing plausible **misinformation**.

Recent benchmarks address these issues by testing LLMs for **logical and common sense reasoning, dataset-specific bias, the ability of models to keep track of information, and downstream tasks without task-specific gradient optimization**. A few examples of such benchmarks are [CoQA](https://stanfordnlp.github.io/coqa/), [LAMBDA](https://zenodo.org/record/2630551#.X4Xzn5NKjUI), [HELLASWAG](https://rowanzellers.com/hellaswag/), [LogiQA](https://github.com/lgw863/LogiQA-dataset). These benchmarks provide methods for evaluating LLMs for mismatches between the behavior we want LLMs to exhibit and the behavior we observe in practice as a result of the training objectives and data we use.

- https://instruction-tuning-with-gpt-4.github.io/: ask GPT-4 to rate LLM response from 1 to 10, User-Oriented-Instructions-252, Vicuna-Instructions-80; [Can Large Language Models Be an Alternative to Human Evaluation?](https://twitter.com/dcml0714/status/1653937790839824384) paper shows ChatGPT rating human-write stories similar to human english teachers, test on 400 stories;
- https://github.com/WeOpenML/PandaLM/: finetune alpaca-52K dataset on several 7B model, using ChatGPT generate 300K comparison score for each 2 response pair and build reward model base on it  
- https://lmsys.org/blog/2023-05-10-leaderboard/: Chatbot Arena leaderboard, Elo ratings of all 13 models, which are based on the 13K anonymous voting data; network constraint in china; [**Directly Comparison with community votes !**]
- https://github.com/ymcui/Chinese-LLaMA-Alpaca/blob/pro/examples/README.md: a chinese benchmark contains 10 tasks (20 samples for each, sum as 200) [**Good start**]
- https://github.com/CLUEbenchmark/SuperCLUE: a chinese benchmark with 10 basic ability, 10 chinese ability, 50+ specialty, but **Not** providing any evaluation detail or dataset or dataset size [**Unreasonable project, not recommended**]
- https://openai.com/research/truthfulqa: TruthfulQA, a OpenAI benchmark for Measuring how models mimic human falsehoods, which comprises 817 questions that span 38 categories, including health, law, finance and politics. [**Special perspective!**]
- https://github.com/EleutherAI/lm-evaluation-harness: 200+ task implemented(hundreds/thousands/ten-thousands for each), support transformers, NeoX, .etc, more detail see this report [Evaluating-Large-Language-Models-LLMs-with-Eleuther-AI](https://wandb.ai/wandb_gen/llm-evaluation/reports/Evaluating-Large-Language-Models-LLMs-with-Eleuther-AI--VmlldzoyOTI0MDQ3) [**Cool! Great Job**]
- https://github.com/google/BIG-bench: collaborative benchmark contains more than 200 tasks [**Sound Great**]
- https://github.com/Hello-SimpleAI/chatgpt-comparison-detection: tens of thousands of comparison responses from both human experts and ChatGPT, with questions ranging from open-domain, financial, medical, legal, and psychological areas. We call the collected dataset the Human ChatGPT Comparison Corpus (HC3).
- https://github.com/hendrycks/test: Multi-task Language Understanding(MMLU), Sep 2020, a multitask accuracy test covers 57 tasks including elementary mathematics, US history, computer science, law, and more, [leaderboard](https://paperswithcode.com/sota/multi-task-language-understanding-on-mmlu), [**good open source but old**]
- https://github.com/Felixgithub2017/MMCU: Multi-task Chinese Understanding(MMCU), Apr 2023, measure the multitask accuracy of Chinese LLMs,  encompasses four major domains, including medicine, law, psychology, and education, with 15 subtasks in medicine and 8 subtasks in education. Email to author for apply free dataset download. [**Seems reasonable**]
- https://github.com/microsoft/AGIEval: MSFT release a benchmark derived from 20 official, public, hign-standard qualification human exams, like gaokao, SAT. AGIEval v1.0 contains 20 tasks, including two cloze tasks (Gaokao-Math-Cloze and MATH) and 18 multi-choice question answering tasks (the rest).
- https://gpt4all.io/index.html: open source project gpt4all benchmark: BoolQ | PIQA | HellaSwag | WinoGrande | ARC-e | ARC-c | OBQA
- https://github.com/bigcode-project/bigcode-evaluation-harness: OpenAI's code-cushman-001, ([tweet](https://twitter.com/LoubnaBenAllal1/status/1655932400541769728))
- https://openai.com/research/evaluating-large-language-models-trained-on-code: OpenAI Codex performance on HumanEval.
- https://github.com/rlancemartin/auto-evaluator: Evaluation tool for LLM QA chains
- https://github.com/Hannibal046/Awesome-LLM/blob/main/paper_list/evaluation.md: a evaluation list, contains traditional NLP tasks
- [GPT-4 Technical Report, OpenAI, 2023.03]
- [Sparks of Artificial General Intelligence: Early experiments with GPT-4, MSFT, 2023.04]
- [PaLM 2 Technical Report, Google, 2023.05]


## Evaluation Dilemma

- Benchmark dataset may have been collected in pre-training/post-training, which make it can't evaluate LLM properly
- LLM could involve with specific data, evaluation only take place on one snapshot
- Too many benchmark to evaluate if you want a comprehensive evaluation
- Generative LLM alway need human evaluation or prompting actions, which cost too much



## Evaluation In a Afforable Way


### Human Exams
- professional language proficiency exams (PaLM2)
  - Chinese: HSK汉语水平考试
  - Japanese: J-Test日本语检定
  - Frech: TCF Test...
  - Spanish: DELE C2...
  - German: Goethe-Zertifikat C2
  - Italian: PLIDA C2...
- professional exams (GPT-4)
  - SAT
  - GRE
  - Medical Knowledge Self-Assesment Program
  - AP Art/Biology/Calculus/Chemistry/English Literature/Environment/Physics/Psychology/Statistics/History/Government...
  - ...
  - Leetcode(Easy/Midium/Hard)
- https://github.com/microsoft/AGIEval: MSFT release a benchmark derived from 20 official, public, hign-standard qualification human exams, like gaokao, SAT. AGIEval v1.0 contains 20 tasks, including two cloze tasks (Gaokao-Math-Cloze and MATH) and 18 multi-choice question answering tasks (the rest).
- ![image](https://github.com/microsoft/AGIEval/raw/main/AGIEval_tasks.png)
- **Recommandation**: Chinese HSK, Gaokao, SAT (https://github.com/microsoft/AGIEval)
- **Purpose**: test for basic language and knowledge understanding
- **Evaluation Method**: zero-shot multiple choice questions(Prompt+Auto), free-text response(Human)
- **Evalution Metrics**: Accuracy/Scores of single/multiple choice questions



### Question Answering and Classification

- English QA and classification tasks(one-shot setting)
  - Open-domain closed-book question answering tasks: TriviaQA (Joshi et al., 2017), Natural Questions2
(Kwiatkowski et al., 2019), and WebQuestions (Berant et al., 2013)
  - Cloze and completion tasks: LAMBADA (Paperno et al., 2016), HellaSwag (Zellers et al., 2019), and StoryCloze
(Mostafazadeh et al., 2016)
  - Winograd-style tasks: Winograd (Levesque et al., 2012) and WinoGrande (Sakaguchi et al., 2021)
  - Reading comprehension: SQuAD v2 (Rajpurkar et al., 2018) and RACE (Lai et al., 2017)
  - Common sense reasoning: PIQA (Bisk et al., 2020), ARC (Clark et al., 2018), and OpenBookQA (Mihaylov
et al., 2018)
  - SuperGLUE (Wang et al., 2019)
  - Natural language inference: Adversarial NLI (ANLI; Nie et al., 2020)
  - ![image](https://github.com/shm007g/LLaMA-Cult-and-More/assets/16999665/73c747ae-6b60-4c07-be0f-8e843e9ac824)
- Multilingual QA (one-shot and no-content setting): TyDi QA (Clark et al., 2020)
- Multilingual toxicity classification
  - Toxicity classification with CivilComments
  - Multilingual toxicity classification with Jigsaw Multilingual
- https://openai.com/research/truthfulqa: TruthfulQA, a OpenAI benchmark for Measuring how models mimic human falsehoods(based on misconceptions and biases they may have), which comprises 817 questions that span 38 categories(with a median of 7 questions and a mean of 21.5 questions per category), including health, law, finance, science and politics. 
- ![image](https://github.com/shm007g/LLaMA-Cult-and-More/assets/16999665/b499ef3d-4ce7-4249-833b-839abbfba651)
- https://yonatanbisk.com/piqa/data/: 20,000 QA pairs that are either multiple-choice or true/false questions, main works on daily physical interaction and common sense reasoning
- ![](https://production-media.paperswithcode.com/datasets/Screen_Shot_2021-03-16_at_2.10.51_PM.png)
- **Recommandation**: TruthfulQA (https://github.com/sylinrl/TruthfulQA) | PIQA (https://leaderboard.allenai.org/physicaliqa/submissions/get-started)
- **Purpose**: QA is about the common test bed, TruthfulQA do falsehood/hallucinations evaluation, PIQA do common sense evaluation and frequently used;
- **Evaluation Method**: BLEURT, GPT-Judge/Human | One-Shot Prompting, Accuracy(Binary Choice)
- ![image](https://github.com/shm007g/LLaMA-Cult-and-More/assets/16999665/c669e9ac-4e69-438a-8a7c-782cf752ce39)



### Reasoning(Common Sense, Math)

- representative reasoning datasets in a few-shot setting: WinoGrande (Sakaguchi et al., 2021), ARC-C (Clark et al., 2018), DROP (Dua et al.,2019), StrategyQA (Geva et al., 2021), CommonsenseQA (CSQA; Talmor et al., 2019), XCOPA (Ponti et al., 2020), and BIG-Bench (BB) Hard (Suzgun et al., 2022).
  - Multilingual common sense reasoning: XCOPA 
  - BIG-Bench (BB) Hard: 23 tasks from 200+, where LLM performed below average human, like multi-step arithmetic problems(multistep_arithmetic)
- Mathematical reasoning
  - MATH (Hendrycks et al., 2021), which contains 12,500 problems from high school competitions in 7 mathematics subject areas
  - GSM8K (Cobbe et al., 2021), a dataset of 8,500 grade school math word problems
  - MGSM (Shi et al., 2023), a multilingual version of GSM8K with translations of a subset of examples into ten typologically diverse languages.
  - ![image](https://github.com/shm007g/LLaMA-Cult-and-More/assets/16999665/26e9ad36-f06b-49cb-93b6-ce674edace6f)
- https://github.com/suzgunmirac/BIG-Bench-Hard: 23 challenging BIG-Bench tasks from 200+ BIG-Bench (https://github.com/google/BIG-bench);
- ![image](https://github.com/shm007g/LLaMA-Cult-and-More/assets/16999665/6376fe94-3558-403e-bb21-8e98c1ae0ad2) 
- **Recommandation**: BIG-Bench-Hard (https://github.com/suzgunmirac/BIG-Bench-Hard, https://github.com/google/BIG-bench/blob/main/bigbench/evaluate_task.py)
- **Purpose**: Evaluate ability of LLM to reason, to combine multiple pieces of information, and to make logical inferences | From Paper2(Early Experiments) analysis, even GPT-4 can do only simple math right now, much arithmetic and calculation mistakes on MATH;
- **Evaluation Method**: chain-of-thought (CoT)
- ![](https://github.com/suzgunmirac/BIG-Bench-Hard/blob/main/figures/bbh-setup.png?raw=true)
- **Evalution Metrics**: Accuracy/Score, Evaluate the truth value of a random Boolean expression consisting of Boolean constants (True, False) and basic Boolean operators (and, or, not).


### Coding

- Code Generation: 3 coding datasets: HumanEval (Chen et al., 2021), MBPP (Austin et al., 2021), and ARCADE (Yin et al., 2022), 
- Multilingual Evaluation: BabelCode (Orlanski et al., 2023) which translates HumanEval into a variety of other programming languages including c++, java, go, haskell and julia.
- Leetcode questions(100 for each level)
- https://github.com/openai/human-eval: HumanEval, a docstring-to-code dataset consisting of 164 coding problems that test various aspects of programming logic and proficiency
- ![](https://pbs.twimg.com/media/FvseneQXoAATLcb?format=jpg&name=900x900)
- **Recommandation**: HumanEval (https://github.com/openai/human-eval), most widely used, Python code;
- **Purpose**: Code language models are among the most economically significant and widely-deployed LLMs today
- **Evaluation Method**: Excution in a robust security sandbox (https://github.com/openai/human-eval, https://github.com/bigcode-project/bigcode-evaluation-harness)
- **Evalution Metrics**: Pass@1, Pass@K


### Translation/Multi-lingual 

- WMT21 Experimental Setup: automatic metric using BLEURT, human metric using Multidimensional Quality Metrics (MQM) with hired professional translators
- **Recommandation**:
- **Purpose**:
- **Evaluation Method**:
- **Evalution Metrics**:


### Natural language generation

- https://github.com/csebuetnlp/xl-sum: XLSum (Hasan et al., 2021), which asks a model to summarize a news article for 44 languages
- WikiLingua (Ladhak et al., 2020), which focuses on generating section headers for step-by-step instructions from WikiHow
- XSum (Narayan et al., 2018), which tasks a model with generating a news article’s first sentence
- https://github.com/ymcui/Chinese-LLaMA-Alpaca/blob/pro/examples/README.md: Chinese-LLaMA-Alpaca released a 200 sample chinese sample for 20 generation task;
- **Recommandation**: XLSum (https://github.com/csebuetnlp/xl-sum) | Chinese-LLaMA-Alpaca-200 (https://github.com/ymcui/Chinese-LLaMA-Alpaca/blob/pro/examples/README.md)
- **Purpose**: Evaluate models’ generation quality, naturalness and smoothness;
- **Evaluation Method**: 1-shot prompting | zero-shot prompting
- **Evalution Metrics**: ROUGE-L; SentencePiece-ROUGE-2, an extension of ROUGE that handles non-Latin characters using a SentencePiece tokenizer | Human Elo rating from 1 to 10

| 测试任务         |                详细样例                | 样例数 | 中文Alpaca-7B | 中文Alpaca-13B | 中文Alpaca-Plus-7B |
| ---------------- | :------------------------------------: | :----: | :-----------: | :------------: | :----------------: |
| **💯总平均分**    |                   -                    |  200   |     65.1      |      70.6      |     **👍🏻75.3**     |
| 知识问答         |            [QA.md](./QA.md)            |   20   |      66       |       74       |      **👍🏻80**      |
| 开放式问答       |           [OQA.md](./OQA.md)           |   20   |   **👍🏻79**    |       74       |      **👍🏻78**      |
| 数值计算、推理   |     [REASONING.md](./REASONING.md)     |   20   |      31       |    **👍🏻50**    |         45         |
| 诗词、文学、哲学 |    [LITERATURE.md](./LITERATURE.md)    |   20   |      68       |       73       |      **👍🏻76**      |
| 音乐、体育、娱乐 | [ENTERTAINMENT.md](./ENTERTAINMENT.md) |   20   |      68       |       74       |      **👍🏻79**      |
| 写信、写文章     |    [GENERATION.md](./GENERATION.md)    |   20   |      76       |    **👍🏻81**    |      **👍🏻81**      |
| 文本翻译         |   [TRANSLATION.md](./TRANSLATION.md)   |   20   |      76       |       78       |      **👍🏻82**      |
| 多轮交互         |      [DIALOGUE.md](./DIALOGUE.md)      |   20   |   **👍🏻83**    |       73       |      **👍🏻84**      |
| 代码编程         |          [CODE.md](./CODE.md)          |   20   |      57       |    **👍🏻64**    |         59         |
| 伦理、拒答       |        [ETHICS.md](./ETHICS.md)        |   20   |      47       |       65       |      **👍🏻89**      |



## LLM Arena (side by side comparison)

- **Purpose**: Arena perform direct comparison while point/scores compare LLM in a indirect way, which gives a clearer preference;
- **Evaluation Method**: Elo rating by online voters
- https://lmsys.org/blog/2023-05-10-leaderboard/
- ![](https://lmsys.org/images/blog/leaderboard_week2/claude_vs_gpt4.png)
- ![](https://lmsys.org/images/blog/leaderboard_week2/english_vs_non_english.png)
