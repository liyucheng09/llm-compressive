import json
from glob import glob
import sys
import os

if __name__ == '__main__':
    # result_path, = sys.argv[1:]
    result_paths = ['/mnt/fast/nobackup/users/yl02706/newac_compressive', '/mnt/fast/nobackup/users/yl02706/compressive']
    # result_paths = ['/mnt/fast/nobackup/users/yl02706/compressive_newcode']
    datasets = ['bbc_image', 'wikitext', 'arxiv', 'code', 'bbc_news', 'math']
    # datasets = ['wikitext']

    for ds in datasets:
        all_results = {}
        for result_path in result_paths:
            path = os.path.join(result_path, f'{ds}/*/*.json')
            results_files = glob(path)

            if len(results_files) == 0:
                print(path)
                print(f'No results found for {ds}')
                continue
        
            for rf in results_files:
                model_name = rf.split('/')[-1][:-5]

                time_stamp = rf.split('/')[-2]
                with open(rf, 'r') as f:
                    results = json.load(f)
                    results = results['ratio']
                
                if model_name in ['LLaMA-7B', 'Llama-2-7B']:
                    continue

                if model_name in all_results:
                    if time_stamp in all_results[model_name]:
                        continue
                else:
                    all_results[model_name] = {}
                
                all_results[model_name][time_stamp] = results
        
        sorted_results = {model: dict(sorted(timestamps.items())) for model, timestamps in sorted(all_results.items())}
        with open(f'results/{ds}_results.json', 'w') as f:
            json.dump(sorted_results, f, indent=2, ensure_ascii=False)
        
        print(f'Finished {ds}, saved to {ds}_results.json')
