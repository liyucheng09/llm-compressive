from fact_qa import FactQA, edit_distance_scorer
import os
import json
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

class CodeCompletion(FactQA):
    def _make_prompt(
        self,
        task, # choose from ['question_generation', 'post_processing', 'answer_generation']
        article = None,
        question = None,
        ref_answer = None,
        last_line = None,
        out = None,
        model_name = None,
    ):
        if task == 'post_processing':
            assert last_line is not None
            
            last_line = last_line.strip()
            prompt = (
                "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{last_line}\n\n"
                "Is the above line real code or just comment? Reply \"Yes\" if it's code, reply \"No\" if it's comment (for example has # or // at the begining). "
                "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            )
            prompt = prompt.format(
                last_line=last_line
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
            assert ref_answer is not None and out is not None

            ref_answer = ref_answer.strip()
            out = out.strip()
            templates = {
                'meta-llama/Meta-Llama-3-8B-Instruct': (
                    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                    "Here are a line of code: the reference code A. \"{ref_answer}\"; and the student code B. \"{out}\". "
                    "Now, your job is to judge whether the student answer (B) reflect the reference answer (A)? "
                    "You should reply \"Yes\" if the student answer make sense according to the reference answer; otherwise reply \"No\". "
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
        
        else:
            raise ValueError(f'Unknown task: {task}')
        
    def make_example(
        self,
        code,
        tokenizer
    ):
        # retrieve a random block from the code (about 100 lines of code)
        # in the block, use the last line as the answer to be generated, the rest as the context

        # block
        all_lines = [i for i in code.split('\n') if i.strip()]
        if len(all_lines) < 120:
            return None
        
        random_block_start = np.random.randint(0, len(all_lines) - 110)
        block = all_lines[random_block_start:random_block_start+100]
        
        # find a last line that is longer than 30 characters
        while True:
            if not block:
                return None
            last_line = block.pop()
            tokenized_last_line = tokenizer.encode(last_line, add_special_tokens=False)
            if len(tokenized_last_line) > 12:
                break
        
        prompt_len = 0
        num_lines_in_prompt = 0
        for i in range(len(block), 0, -1):
            i-=1
            line = block[i]
            prompt_len += len(tokenizer.encode(line, add_special_tokens=False))
            if prompt_len > 2048:
                break
            num_lines_in_prompt += 1

        block = block[-num_lines_in_prompt:]
        block = '\n'.join(block)

        # add a small prompt from the last line to the block
        prompting_tokens = tokenizer.decode(tokenized_last_line[:3])
        left_tokens = tokenizer.decode(tokenized_last_line[3:])

        return {
            'code': block,
            'last_line': last_line,
            'prompting_tokens': prompting_tokens,
            'left_tokens': left_tokens
        }
    
    def make_test(
        self
    ):
        tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf', use_fast=True)
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
            for example in tqdm(dataset):
                code = example['code']
                example = self.make_example(code, tokenizer)

                if example is None:
                    continue
                
                count += 1
                self.test_set[time].append(example)

                if count >= self.num_examples_per_time:
                    break

                if count % 50 == 0:
                    self.save_to_file(
                        task='question_generation',
                        time=time
                    )

            self.save_to_file(
                task='question_generation',
                time=time
            )

        print('Finished making test set.')

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
            
            blocks = [example['code'] for example in test_set if example is not None]
            prompting = [example['prompting_tokens'] for example in test_set if example is not None]
            last_lines = [example['last_line'] for example in test_set if example is not None]
            prompts = [example['code'] + '\n' + example['prompting_tokens'] for example in test_set if example is not None]
            ref_answers = [example['left_tokens'] for example in test_set if example is not None]

            outs = self._inference_with_vllm(
                task='answer_generation',
                prompts=prompts,
                model_name=model_name
            )

            for block, left_tokens, last_line, prompting_tokens, out in zip(blocks, ref_answers, last_lines, prompting, outs):
                self.answer_set[time].append({
                    'code': block,
                    'prompt': out.prompt,
                    'prompting_tokens': prompting_tokens,
                    'last_line': last_line,
                    'left_tokens': left_tokens,
                    'out': [o.text for o in out.outputs],
                })

            self.save_to_file(
                task='answer_generation',
                time=time
            )
        
        print('Finished getting answers.')
    
    def postprocess(
        self,
        model_name: str
    ):
        for time, test_set in tqdm(self.test_set.items(), desc='loop over months'):
            # skip?
            if self.do_skip:
                f = os.path.join(self.postprocessed_output_dir, f'{time}.json')
                if os.path.exists(f):
                    with open(f, 'r') as f:
                        self.test_set[time] = json.load(f)
                    continue

            finalized_test_set = []
            prompts = []
            for index, example in tqdm(enumerate(test_set), desc='loop over examples'):
                last_line = example['last_line']
                prompt = self._make_prompt(task='post_processing', last_line=last_line)
                prompts.append(prompt)

            outs = self._inference_with_vllm(
                task='post_processing',
                prompts=prompts,
                model_name=model_name
            )

            for example, out in zip(test_set, outs):
                response = out.outputs[0].text

                answerable = self._parse_response(
                    task='post_processing',
                    response=response
                )

                if answerable:
                    finalized_test_set.append(example)

            self.test_set[time] = finalized_test_set
            self.save_to_file(
                task='post_processing',
                time=time
            )

        print('Finished postprocessing test set.')
    
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

                last_line = example['last_line']
                prompting_tokens = example['prompting_tokens']
                example['out'] = [ i.split('\n', 1)[0] for i in example['out']]

                prompts = [
                    self._make_prompt(
                        task='llm_scorer',
                        ref_answer=last_line,
                        out= prompting_tokens + o,
                        model_name=model_name
                    )
                    for o in example['out']
                ]

                responses = self._inference_with_vllm(
                    task='llm_scorer',
                    prompts=prompts,
                    model_name=model_name
                )

                auto_acc = edit_distance_scorer(
                    example['out'],
                    example['left_tokens']
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
                example['auto_acc'] = auto_acc

            self.save_to_file(
                task='llm_scorer',
                time=time
            )

        print('Finished getting answers.')

if __name__ == '__main__':

    times = [f'{year}-{month:02}' for year in range(2017, 2025) for month in range(1, 13)]
    times = times[:86]

    cc = CodeCompletion(
        task_name='code_completion',
        dataset_name='RealTimeData/code_alltime',
        times=times,
        data_dir='real_tasks/cc',
        output_dir='real_tasks/cc/cc_llama_2_7b_chat',
        num_examples_per_time=400,
        do_skip=False
    )

    cc.make_test()
    cc.postprocess(model_name='meta-llama/Meta-Llama-3-8B-Instruct')
    # cc.get_answer(model_name='meta-llama/Llama-2-7b-hf')
    cc.get_answer(model_name='meta-llama/Llama-2-7b-chat-hf')
    cc.llm_scorer(model_name='meta-llama/Meta-Llama-3-8B-Instruct')