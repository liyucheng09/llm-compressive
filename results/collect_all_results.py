import json
from glob import glob
import sys
import os

if __name__ == '__main__':
    result_path, = sys.argv[1:]
    datasets = ['bbc_news', 'wikitext', 'bbc_image', 'code']

    for ds in datasets:
        all_results = {}
        path = os.path.join(result_path, f'{ds}/*/*.json')
        results_files = glob(path)

        if len(results_files) == 0:
            print(path)
            print(f'No results found for {ds}')
            continue

        for rf in results_files:
            model_name = rf.split('/')[-1].split('.')[0]
            time_stamp = rf.split('/')[-2]
            with open(rf, 'r') as f:
                results = json.load(f)
            
            if model_name not in all_results:
                all_results[model_name] = {}
            
            all_results[model_name][time_stamp] = results
        
        sorted_results = {model: dict(sorted(timestamps.items())) for model, timestamps in sorted(all_results.items())}
        with open(f'{ds}_results.json', 'w') as f:
            json.dump(sorted_results, f, indent=2, ensure_ascii=False)
        
        print(f'Finished {ds}, saved to {ds}_results.json')
