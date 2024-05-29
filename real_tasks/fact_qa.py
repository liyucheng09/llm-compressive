import datasets
import os
from openai import OpenAI
import json
from tqdm import tqdm
from datetime import datetime
import numpy as np
from glob import glob

import Levenshtein
from transformers import AutoTokenizer

def edit_distance_scorer(
    gen_answers,
    ref_answer
):
    return (len(ref_answer) - np.mean([Levenshtein.distance(ref_answer.lower().strip(), o.lower().strip()) for o in gen_answers])) / len(ref_answer)

class FactQA:
    def __init__(
        self,
        task_name: str,
        dataset_name,
        times: list[str],
        data_dir: str,
        output_dir: str,
        num_examples_per_time: int = 100,
        do_skip: bool = True
    ):
        self.datasets = {}
        self.test_set = {}

        self.question_output_dir = os.path.join(data_dir, task_name)
        self.postprocessed_output_dir = os.path.join(data_dir, f'{task_name}_filtered')
        self.answer_output_dir = os.path.join(output_dir, f'{task_name}_answer')
        self.scored_output_dir = os.path.join(output_dir, f'{task_name}_scored')

        os.makedirs(self.question_output_dir, exist_ok=True)
        os.makedirs(self.postprocessed_output_dir, exist_ok=True)
        os.makedirs(self.answer_output_dir, exist_ok=True)
        os.makedirs(self.scored_output_dir, exist_ok=True)

        for time in times:
            self.datasets[time] = datasets.load_dataset(dataset_name, time, split='train').shuffle(seed=42)

        self.do_skip = do_skip
        self.num_examples_per_time = num_examples_per_time
        openai_key = os.environ.get('OPENAI_API_KEY', None)
        if openai_key is not None:
            self.llm = OpenAI(
                api_key=openai_key,
                base_url='https://api.deepinfra.com/v1/openai',
            )
        else:
            def need_key(*args, **kwargs):
                raise ValueError("Please provide OpenAI API key.")
            self.llm = need_key
    
    def _init_vllm(
        self,
        model_name: str,
    ):
        from vllm import LLM, SamplingParams
        self.sampling_params = {
            "question_generation": SamplingParams(temperature=1, top_p=0.95, max_tokens=256, n=1),
            "answer_generation": SamplingParams(temperature=1, top_p=0.95, max_tokens=100, n=10, use_beam_search=False),
            "post_processing": SamplingParams(temperature=0, n=1, max_tokens=8),
            "llm_scorer": SamplingParams(temperature=0, n=1, max_tokens=8),
        }
        self.vllm = LLM(model=model_name)
    
    def _inference_with_vllm(
        self,
        task: str,
        prompts: str,
        model_name: str,
    ):
        if not hasattr(self, 'vllm'):
            self._init_vllm(model_name)

        return self.vllm.generate(prompts, self.sampling_params[task])

    def save_to_file(
        self,
        task, # choose from ['question_generation', 'post_processing', 'answer_generation']
        time,
    ):
        if task == 'question_generation':
            with open(os.path.join(self.question_output_dir, f'{time}.json'), 'w') as f:
                json.dump(self.test_set[time], f, indent=2, ensure_ascii=False)
        elif task == 'post_processing':
            with open(os.path.join(self.postprocessed_output_dir, f'{time}.json'), 'w') as f:
                json.dump(self.test_set[time], f, indent=2, ensure_ascii=False)
        elif task == 'answer_generation':
            with open(os.path.join(self.answer_output_dir, f'{time}.json'), 'w') as f:
                json.dump(self.answer_set[time], f, indent=2, ensure_ascii=False)
        elif task == 'llm_scorer':
            with open(os.path.join(self.scored_output_dir, f'{time}.json'), 'w') as f:
                json.dump(self.answer_set[time], f, indent=2, ensure_ascii=False)

    def _make_llm_request(
        self,
        prompt
    ):
        response = self.llm.chat.completions.create(
            model="mistralai/Mixtral-8x22B-Instruct-v0.1",
            messages=[{"role": "user", "content": prompt}],
        )

        return response.choices[0].message.content
    
    def _chunked_text(
        self,
        text,
        tokenizer,
        max_len=3072,
    ):
        if len(text) > max_len * 10:
            text = text[:max_len * 10]
        
        tokenized_text = tokenizer(text, add_special_tokens=False)
        chunked = tokenized_text['input_ids'][:max_len]

        return tokenizer.decode(chunked)
            
    def _make_prompt(
        self,
        task, # choose from ['question_generation', 'post_processing', 'answer_generation']
        article = None,
        question = None,
        ref_answer = None,
        out = None,
        model_name = None,
    ):
        if task == 'question_generation':
            assert article is not None

            prompt = (
                "Title: {title}\n\nDate: {published_date}\nTLDR\n{description}\nMain Context:\n{content}\n\n========\n\nWhat's the key information in this article? I am making some quiz questions with short and quick answers. "
                "Help me find the key information, including but not limited to who, what, where, and when. For each key piece of information, ask a corresponding question next to the answer. Returns up to three QA pairs as a json dictionary list."
            )
            prompt = prompt.format(
                title=article['title'],
                published_date=article['published_date'],
                description=article['description'],
                content=article['content']
            )

            demonstration = """[
{
    "question": "Who has Stoke City signed?",
    "answer": "Saido Berahino"
},
{
    "question": "From which club has Stoke City signed Saido Berahino?",
    "answer": "West Brom"
},
{
    "question": "What is the fee Stoke City paid for Saido Berahino?",
    "answer": "£12m"
}
]"""
            prompt += "Here is a demonstration of how your output should look like:\n" + demonstration

            return prompt
        
        elif task == 'post_processing':
            assert question is not None

            prompt = (
                # "Is the following quiz question answerable? or they require more context? Some example of unanswerable question: What was the score of the match?/How many family members are there in this Yorkshire family? Some answerable questions: Why is Martin McGuinness retiring from frontline politics?/Where is Saffron Jackson from? You should answer \"Yes\" or \"No\" as your answer, and nothing else. Question: {question}"
                "Is the following quiz question answerable? You should answer \"Yes\" or \"No\" as your answer, and nothing else. Unless significant information is missing in the question, you should reply \"yes\". Question: {question}"
            )
            prompt = prompt.format(
                question=question
            )

            return prompt

        elif task == 'answer_generation':
            templates = {
                'meta-llama/Llama-2-7b-hf': "Question: {question}\nAnswer: ",
                'meta-llama/Llama-2-7b-chat-hf': "[INST] {question} [/INST]",
                'meta-llama/Meta-Llama-3-8B': "Question: {question}\nAnswer: ",
                'meta-llama/Meta-Llama-3-8B-Instruct': "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            }
            model_name = model_name if model_name is not None else 'meta-llama/Llama-2-7b-hf'

            prompt = templates[model_name].format(
                question=question
            )

            return prompt
        
        elif task == 'llm_scorer':
            templates = {
                'meta-llama/Meta-Llama-3-8B-Instruct': (
                    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                    "Here is a question: \"{question}\". And its reference answer is \"{ref_answer}\". "
                    "Now, your job is to judge whether the following student answer contains the reference answer? "
                    "Student answer: {out} "
                    "You should reply \"Yes\" if the student answer includes reference answer; otherwise reply \"No\". "
                    "Return only \"Yes\" or \"No\" as your judgement and return nothing else. "
                    "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                ),
                'gpt-4': (
                    "The question is: {question}. And its reference answer is {ref_answer}. "
                    "Now, your job is to judge whether the following answer is correct? Student answer: {out} "
                    "You should only reply \"Yes\" or \"No\" as your answer and return nothing else."
                )
            }

            prompt = templates[model_name].format(
                question=question,
                ref_answer=ref_answer,
                out=out
            )

            return prompt

    def _parse_response(
        self,
        task, # choose from ['question_generation', 'post_processing', 'answer_generation']
        response
    ):
        if task == 'question_generation':
            try:
                response = eval(response)
            except:
                response = None

            return response
        
        elif task == 'post_processing':

            if 'yes' in response.lower():
                return True
            elif 'no' in response.lower():
                return False
            else:
                return None

        elif task == 'answer_generation':
            raise NotImplementedError

    def make_test(
        self,
    ):
        for time, dataset in tqdm(self.datasets.items(), desc='loop over months'):
            # skip?
            if self.do_skip:
                f = os.path.join(self.question_output_dir, f'{time}.json')
                if os.path.exists(f):
                    with open(f, 'r') as f:
                        self.test_set[time] = json.load(f)
                    continue

            count = 0
            if time not in self.test_set:
                self.test_set[time] = []
            for article in tqdm(dataset):
                prompt = self._make_prompt(task='question_generation', article=article)
                response = self._make_llm_request(prompt)
                response = self._parse_response(
                    task='question_generation',
                    response=response
                )
                if response is not None:
                    count += 1

                self.test_set[time].append({
                    'prompt': prompt,
                    'article': article,
                    'response': response
                })

                if count >= self.num_examples_per_time:
                    break

                if count % 10 == 0:
                    self.save_to_file(
                        task='question_generation',
                        time=time
                    )

            self.save_to_file(
                task='question_generation',
                time=time
            )

        print('Finished making test set.')
    
    def postprocess(
        self,
    ):
        for time, test_set in tqdm(self.test_set.items(), desc='loop over months'):
            # skip?
            if self.do_skip:
                f = os.path.join(self.postprocessed_output_dir, f'{time}.json')
                if os.path.exists(f):
                    with open(f, 'r') as f:
                        self.test_set[time] = json.load(f)
                    continue

            for index, example in tqdm(enumerate(test_set), desc='loop over examples'):
                if example['response'] is None:
                    continue

                finalized_test_set = []
                date = datetime.strptime(example['article']['published_date'], "%Y-%m-%d")
                date_str = date.strftime("%b %Y")
                for question in example['response']:
                    try:
                        question['question'] = f'{date_str}: {question["question"]}'
                    except:
                        continue
                    prompt = self._make_prompt(task='post_processing', question=question['question'])
                    response = self._make_llm_request(prompt)
                    answerable = self._parse_response(
                        task='post_processing',
                        response=response
                    )

                    if answerable:
                        finalized_test_set.append(question)

                example['filtered_qa'] = finalized_test_set

                if index % 10 == 0:
                    self.save_to_file(
                        task='post_processing',
                        time=time
                    )
                
            self.save_to_file(
                task='post_processing',
                time=time
            )

        print('Finished postprocessing test set.')
    
    def get_answer(
        self,
        model_name: str
    ):
        self.answer_set = {}
        
        for time, test_set in tqdm(self.test_set.items(), desc='loop over months'):
            # skip?
            if self.do_skip:
                f = os.path.join(self.answer_output_dir, f'{time}.json')
                if os.path.exists(f):
                    with open(f, 'r') as f:
                        self.answer_set[time] = json.load(f)
                    continue
            
            if time not in self.answer_set:
                self.answer_set[time] = []
            
            questions = []
            ref_answers = []
            article_links = []
            prompts = []
            for example in test_set:
                if 'filtered_qa' not in example or example['filtered_qa'] is None:
                    continue
                
                for qa in example['filtered_qa']:
                    try:
                        time_s, query = qa['question'].split(': ')
                    except:
                        print('Wrong format')
                        print(qa['question'])
                        continue
                    query = f"In {time_s}, {query}"
                    prompts.append(self._make_prompt(
                        task='answer_generation',
                        question=query,
                        model_name=model_name
                    ))
                    questions.append(query)
                    ref_answers.append(qa['answer'])
                    article_links.append(example['article']['link'])

            outs = self._inference_with_vllm(
                task='answer_generation',
                prompts=prompts,
                # model_name='meta-llama/Llama-2-7b-hf'
                model_name=model_name,
            )

            for question, ref_answer, out, link in zip(questions, ref_answers, outs, article_links):
                gen_answers = [o.text for o in out.outputs]
                self.answer_set[time].append({
                    'question': question,
                    'ref_answer': ref_answer,
                    'prompt': out.prompt,
                    'out': gen_answers,
                    'link': link,
                    'accuracy': self.auto_scorer(gen_answers, ref_answer),
                })

            self.save_to_file(
                task='answer_generation',
                time=time
            )
    
    def auto_scorer(
        self,
        outs,
        ref_answer
    ):
        acc = np.mean([1 if ref_answer.lower() in o.lower() else 0 for o in outs])
        return acc

    def llm_scorer(
        self,
        model_name,
    ):
        for time, answer_set in tqdm(self.answer_set.items(), desc='loop over months'):
            # skip?
            if self.do_skip:
                f = os.path.join(self.scored_output_dir, f'{time}.json')
                if os.path.exists(f):
                    with open(f, 'r') as f:
                        self.answer_set[time] = json.load(f)
                    continue

            for example in answer_set:
                ref_answer = example['ref_answer']
                query = example['question']

                prompts = [
                    self._make_prompt(
                        task='llm_scorer',
                        question=query,
                        ref_answer=ref_answer,
                        out=o if '\nQuestion' not in o else o.split('\nQuestion')[0],
                        model_name=model_name
                    )
                    for o in example['out']
                ]

                responses = self._inference_with_vllm(
                    task='llm_scorer',
                    prompts=prompts,
                    model_name=model_name
                )

                num_correct = 0
                num_wrong = 0
                judges = []
                for r in responses:
                    judges.append(r.outputs[0].text)
                    if 'yes' in r.outputs[0].text.lower():
                        num_correct += 1
                    elif 'no' in r.outputs[0].text.lower():
                        num_wrong += 1
                
                if num_correct + num_wrong == 0:
                    example['llm_acc'] = None
                else:
                    example['llm_acc'] = num_correct / (num_correct + num_wrong)
                example['judges'] = judges

            self.save_to_file(
                task='llm_scorer',
                time=time
            )

        print('Finished getting answers.')

def create_batch_request_to_OpenAI(
    path,
    output_path,
    tasks,
):
    os.makedirs(output_path, exist_ok=True)

    tempates = {
        'llm_scorer': (
            "Here is a question: \"{question}\". And its reference answer is \"{ref_answer}\". "
            "Now, your job is to judge whether the following answer is correct? Student answer: \"{out}\". "
            "You should reply \"Yes\" or \"No\" as your judgement and return nothing else."
        )
    }

    files = glob(os.path.join(path, '*.json'))
    requests = []
    for f in files:
        time = f.split('/')[-1].split('.')[0]
        output_file = os.path.join(output_path, f'{time}.json')
        if os.path.exists(output_file):
            continue

        with open(f, 'r') as f:
            data = json.load(f)

        if tasks == 'llm_scorer':
            for e_idx, example in enumerate(data):
                for o_idx, o in enumerate(example['out']):
                    # time, query = example['question'].split(':', 1)
                    # query = f"In {time}, {query}"

                    if '\nQuestion' in o:
                        o = o.split('\nQuestion')[0]
                    # if len(o) > 2 * len(example['ref_answer']):
                    #     fo = ''
                    #     for i in o.split('\n'):
                    #         if not i.strip():
                    #             continue
                    #         if len(fo) + len(i) < 2 * len(example['ref_answer']):
                    #             fo += i + '\n'
                    #     o = fo
                    prompt = tempates[tasks].format(
                        question=example['question'],
                        ref_answer=example['ref_answer'],
                        out=o
                    )

                    requests.append(
                        {
                            "custom_id": f"{time}-{e_idx}-{o_idx}",
                            "method": "POST",
                            "url": "/v1/chat/completions",
                            "body": {
                                "model": "gpt-3.5-turbo-0125",
                                "messages": [{"role": "user", "content": prompt}],
                                "max_tokens": 8
                            }
                        }
                    )

    # write requests to jsonl
    with open(os.path.join(output_path, 'requests.jsonl'), 'w') as f:
        for r in requests:
            f.write(json.dumps(r) + '\n')
    
    client = OpenAI(
        api_key=os.environ.get('OPENAI_API_KEY'),
    )
    batch_input_file = client.files.create(
        file=open("real_tasks/data/factqa_scored_openai/requests.jsonl", "rb"),
        purpose="batch"
    )

    batch_input_file_id = batch_input_file.id

    b = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": "factqa eval jobs"
        }
    )

    print('file_id')
    print(batch_input_file_id)
    print('batch_meta_info')
    print(b)

if __name__ == '__main__':
    # every months from 2017-01 to 2024-02
    times = [f'{year}-{month:02}' for year in range(2017, 2025) for month in range(1, 13)]
    times = times[:86]

    factqa = FactQA(
        task_name='factqa',
        dataset_name='RealTimeData/bbc_news_alltime',
        times=times,
        data_dir='real_tasks/data',
        output_dir='real_tasks/Llama-2-chat',
        num_examples_per_time=100
    )

    factqa.make_test()
    factqa.postprocess()
    # factqa.get_answer(model_name='meta-llama/Llama-2-7b-hf')
    factqa.get_answer(model_name='meta-llama/Llama-2-7b-chat-hf')
    # factqa.get_answer(model_name='meta-llama/Meta-Llama-3-8B')
    factqa.llm_scorer(model_name='meta-llama/Meta-Llama-3-8B-Instruct')

    # create_batch_request_to_OpenAI(
    #     path='real_tasks/data/factqa_scored',
    #     output_path='real_tasks/data/factqa_scored_openai',
    #     tasks='llm_scorer'
    # )