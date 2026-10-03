from lmms_eval.tasks.videomme.utils import videomme_doc_to_text, videomme_process_results, videomme_aggregate_results
import datasets

def videomme_process_docs_gt_none(dataset: datasets.Dataset) -> datasets.Dataset:
    letters = ['A', 'B', 'C', 'D']

    def replace_gt_with_none(example):
        options = example['options']
        answer = example['answer']
        answer_idx = ord(answer.upper()) - ord('A')
        contents = []
        for opt in options:
            if '. ' in opt:
                contents.append(opt.split('. ', 1)[1])
            else:
                contents.append(opt)
        if 0 <= answer_idx < len(contents):
            contents[answer_idx] = 'None'
        new_options = [f'{letters[i]}. {contents[i]}' for i in range(len(contents))]
        return {'options': new_options, 'answer': answer}
    return dataset.map(replace_gt_with_none)
