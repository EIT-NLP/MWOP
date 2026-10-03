from dataclasses import dataclass
from typing import Literal, Optional, Tuple
from transformers import PreTrainedModel, PreTrainedTokenizer

@dataclass
class ChatMlSpecialTokens:
    bos_token: str = '<|im_start|>'
    eos_token: str = '<|im_end|>'
    pad_token: str = '<|im_end|>'

    @property
    def system(self):
        return f'{self.bos_token}system'

    @property
    def user(self):
        return f'{self.bos_token}user'

    @property
    def assistant(self):
        return f'{self.bos_token}assistant'

    @property
    def chat_template(self):
        return f"{{% for message in messages %}}{{{{'{self.bos_token}' + message['role'] + '\n' + message['content'] + '{self.eos_token}' + '\n'}}}}{{% endfor %}}{{% if add_generation_prompt %}}{{{{ '{self.assistant}\n' }}}}{{% endif %}}"
FORMAT_MAPPING = {'chatml': ChatMlSpecialTokens}

def setup_chat_format(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, format: Optional[Literal['chatml']]='chatml', resize_to_multiple_of: Optional[int]=None) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    if format not in FORMAT_MAPPING:
        raise ValueError(f'Format {format} not available. Please use one of {FORMAT_MAPPING.keys()}')
    chat_format = FORMAT_MAPPING[format]()
    tokenizer.eos_token = chat_format.eos_token
    tokenizer.pad_token = chat_format.pad_token
    tokenizer.bos_token = chat_format.bos_token
    tokenizer.add_special_tokens({'additional_special_tokens': [chat_format.bos_token, chat_format.eos_token]})
    tokenizer.chat_template = chat_format.chat_template
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=resize_to_multiple_of if resize_to_multiple_of is not None else None)
    if getattr(model, 'generation_config', None) is not None:
        model.generation_config.bos_token_id = tokenizer.bos_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    return (model, tokenizer)
