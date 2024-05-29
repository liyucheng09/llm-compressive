from fact_qa import *

class Summization(FactQA):
    def make_example(
        self,
        article,
    ):
        pass
    
    def make_test(self):
        for time, dataset in tqdm(self.datasets.items(), desc='Make Test - loop over months'):
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