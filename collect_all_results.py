import json
from glob import glob

if __name__ == '__main__':
    datasets = ['bbc_news', 'wikitext', 'bbc_image']

    for ds in datasets:
        all_results = {}
        results_files = glob(f'/mnt/fast/nobackup/users/yl02706/compressive/{ds}/*/*.json')
        for rf in results_files:
            model_name = rf.split('/')[-1].split('.')[0]
            with open(rf, 'r') as f:
                results = json.load(f)
            all_results[model_name] = results
        with open(f'{ds}_results.json', 'w') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
