from lmms_eval.tasks.videomme.utils import videomme_doc_to_text, videomme_process_results, videomme_aggregate_results
import random
import datasets

def videomme_process_docs_random_choice(dataset: datasets.Dataset) -> datasets.Dataset:

    def shuffle_options(example):
        options = example['options']
        answer = example['answer']
        letters = ['A', 'B', 'C', 'D']
        contents = []
        for opt in options:
            if '. ' in opt:
                contents.append(opt.split('. ', 1)[1])
            else:
                contents.append(opt)
        answer_idx = letters.index(answer.upper()) if answer.upper() in letters else 0
        answer_content = contents[answer_idx]
        seed = hash(example.get('question_id', str(example))) % 2 ** 32
        rng = random.Random(seed)
        indices = list(range(len(contents)))
        rng.shuffle(indices)
        new_contents = [contents[i] for i in indices]
        new_answer_idx = new_contents.index(answer_content)
        new_answer = letters[new_answer_idx]
        new_options = [f'{letters[i]}. {new_contents[i]}' for i in range(len(new_contents))]
        return {'options': new_options, 'answer': new_answer}
    return dataset.map(shuffle_options)
