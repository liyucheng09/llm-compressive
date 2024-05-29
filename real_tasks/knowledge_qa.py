from fact_qa import *

class KnowledgeQA(FactQA):
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
                "Title: {title}\n\nMain Context:\n{content}\n\n========\n\nWhat's the key but rare knowledge in this article? I am making some quiz questions with short and quick answers. "
                "Help me find the key and rare knowledge in this article, but try to focus on information that is simple and concise (so the answer can be as short as possible). "
                "For each key piece of information, ask a corresponding question next to the answer. Remember, the question should include enough context to make the question answerable. "
                "Returns up to five QA pairs as a json dictionary list."
            )
            prompt = prompt.format(
                title=article['title'],
                content=article['content']
            )

            demonstration = """[
{
"question": "Who was Victoria's first cousin whom she married in 1840?",
"answer": "Her first cousin, Prince Albert of Saxe-Coburg and Gotha."
},
{
"question": "What was Victoria's nickname due to her children's marriages into various European royal families?",
"answer": "'The grandmother of Europe'."
},
{
"question": "How many children did Victoria and Albert have together?",
"answer": "Nine children."
},
{
"question": "What title did Victoria adopt in addition to Queen of the United Kingdom in 1876?",
"answer": "Empress of India."
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
                    "Student answer is: \"{out}\" "
                    "You should reply \"Yes\" if the student answer includes reference answer; otherwise reply \"No\". "
                    "Return only \"Yes\" or \"No\" as your judgement and return nothing else. "
                    "<|eot_id|><|start_header_id|>assistant<|end_header_id|> My answer is: "
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

    def make_test(
        self,
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
            for article in tqdm(dataset):
                text = article['text']

                chunked_text = self._chunked_text(text, tokenizer, max_len=3072)
                prompt = self._make_prompt(task='question_generation', article={'title': article['title'], 'content': chunked_text})
                response = self._make_llm_request(prompt)
                response = self._parse_response(
                    task='question_generation',
                    response=response
                )
                if response is not None:
                    count += 1

                self.test_set[time].append({
                    'title': article['title'],
                    'prompt': prompt,
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
            titles = []
            prompts = []
            for example in test_set:
                if example['response'] is None:
                    continue
                
                title = example['title']
                for qa in example['response']:
                    try:
                        query = qa['question']
                        ref_answer = qa['answer']
                    except:
                        continue

                    prompts.append(self._make_prompt(
                        task='answer_generation',
                        question=query,
                        model_name=model_name
                    ))
                    questions.append(query)
                    ref_answers.append(ref_answer)
                    titles.append(title)

            outs = self._inference_with_vllm(
                task='answer_generation',
                prompts=prompts,
                # model_name='meta-llama/Llama-2-7b-hf'
                model_name=model_name,
            )

            for question, ref_answer, out, title in zip(questions, ref_answers, outs, titles):
                gen_answers = [o.text for o in out.outputs]
                self.answer_set[time].append({
                    'question': question,
                    'ref_answer': ref_answer,
                    'prompt': out.prompt,
                    'out': gen_answers,
                    'title': title,
                    'accuracy': self.auto_scorer(gen_answers, ref_answer),
                })

            self.save_to_file(
                task='answer_generation',
                time=time
            )
    
    def llm_scorer(
        self,
        model_name,
    ):
        for time, answer_set in tqdm(self.answer_set.items(), desc='loop over months'):
            # skip?
            # if self.do_skip:
            #     f = os.path.join(self.scored_output_dir, f'{time}.json')
            #     if os.path.exists(f):
            #         with open(f, 'r') as f:
            #             self.answer_set[time] = json.load(f)
            #         continue

            for example in answer_set:
                ref_answer = example['ref_answer']
                query = example['question']

                prompts = [
                    self._make_prompt(
                        task='llm_scorer',
                        question=query,
                        ref_answer=ref_answer,
                        out=o.split('\n')[0],
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
                    tr = r.outputs[0].text
                    judges.append(tr.strip())
                    if 'yes' in tr.lower():
                        num_correct += 1
                    elif 'no' in tr.lower():
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

if __name__ == '__main__':
    # every months from 2017-01 to 2024-02
    times = [f'{year}-{month:02}' for year in range(2017, 2025) for month in range(1, 13)]
    times = times[:86]

    factqa = KnowledgeQA(
        task_name='knowledge_qa',
        dataset_name='RealTimeData/wikitext_alltime',
        times=times,
        data_dir='real_tasks/knowledge_qa',
        output_dir='real_tasks/knowledge_qa/knowledge_qa_llama_2_7b_chat',
        num_examples_per_time=50,
        # do_skip=False
    )

    factqa.make_test()
    # factqa.postprocess()
    # factqa.get_answer(model_name='meta-llama/Llama-2-7b-hf')
    factqa.get_answer(model_name='meta-llama/Llama-2-7b-chat-hf')
    # # factqa.get_answer(model_name='meta-llama/Meta-Llama-3-8B')
    factqa.llm_scorer(model_name='meta-llama/Meta-Llama-3-8B-Instruct')