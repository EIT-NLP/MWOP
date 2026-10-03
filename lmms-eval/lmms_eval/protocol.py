import os
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
import numpy as np
from PIL import Image
from pydantic import BaseModel
from lmms_eval.imports import optional_import
from lmms_eval.models.model_utils.media_encoder import encode_image_to_base64
VideoReader, _has_decord = optional_import('decord', 'VideoReader')
cpu, _ = optional_import('decord', 'cpu')

class ChatTextContent(BaseModel):
    type: Literal['text'] = 'text'
    text: str

class ChatImageContent(BaseModel):
    type: Literal['image'] = 'image'
    url: Any

class ChatVideoContent(BaseModel):
    type: Literal['video'] = 'video'
    url: Any

class ChatAudioContent(BaseModel):
    type: Literal['audio'] = 'audio'
    url: Any
ChatContent = Union[ChatTextContent, ChatImageContent, ChatVideoContent, ChatAudioContent]

class ChatMessage(BaseModel):
    role: Literal['user', 'system', 'assistant']
    content: List[ChatContent]

class ChatMessages(BaseModel):
    messages: List[ChatMessage]

    def extract_media(self):
        images = []
        videos = []
        audios = []
        for message in self.messages:
            for content in message.content:
                if content.type == 'image':
                    images.append(content.url)
                elif content.type == 'video':
                    videos.append(content.url)
                elif content.type == 'audio':
                    audios.append(content.url)
        return (images, videos, audios)

    def to_hf_messages(self, video_kwargs: Optional[Dict[str, str]]=None):
        if video_kwargs is None:
            video_kwargs = {}
        _num_frames = video_kwargs.get('nframes', 32)
        hf_messages = []
        for message in self.messages:
            hf_message = {'role': message.role, 'content': []}
            for content in message.content:
                if content.type == 'text':
                    hf_message['content'].append({'type': 'text', 'text': content.text})
                elif content.type == 'image':
                    hf_message['content'].append({'type': 'image', 'image': content.url})
                elif content.type == 'video':
                    hf_message['content'].append({'type': 'video', 'video': content.url, **video_kwargs})
                elif content.type == 'audio':
                    hf_message['content'].append({'type': 'audio', 'audio': content.url})
            hf_messages.append(hf_message)
        return hf_messages

    def to_openai_messages(self, video_kwargs: Optional[Dict[str, str]]=None):
        if video_kwargs is None:
            video_kwargs = {}
        openai_messages = []
        encode_cache: Dict[Tuple[object, ...], str] = {}
        image_format = os.getenv('LMMS_IMAGE_ENCODE_FORMAT', 'PNG').upper()
        mime_type = f"image/{('jpeg' if image_format == 'JPG' else image_format.lower())}"
        quality = int(os.getenv('LMMS_IMAGE_JPEG_QUALITY', '85')) if image_format in {'JPEG', 'JPG', 'WEBP'} else None
        for message in self.messages:
            openai_message = {'role': message.role, 'content': []}
            for content in message.content:
                if content.type == 'text':
                    openai_message['content'].append({'type': 'text', 'text': content.text})
                elif content.type == 'image':
                    openai_message['content'].append({'type': 'image_url', 'image_url': {'url': f'data:{mime_type};base64,{self.encode_image(content.url, encode_cache, image_format, quality)}'}})
                elif content.type == 'video':
                    if VideoReader is None:
                        raise ImportError('Install decord for video conversion')
                    reader = VideoReader(content.url, ctx=cpu(0))
                    indices = np.linspace(0, len(reader) - 1, int(video_kwargs.get('nframes', 32)), dtype=int)
                    for frame in reader.get_batch(indices).asnumpy():
                        image = Image.fromarray(frame)
                        openai_message['content'].append({'type': 'image_url', 'image_url': {'url': f'data:{mime_type};base64,{self.encode_image(image, encode_cache, image_format, quality)}'}})
                elif content.type == 'audio':
                    openai_message['content'].append({'type': 'audio_url', 'audio_url': {'url': content.url}})
            openai_messages.append(openai_message)
        return openai_messages

    def _calculate_timestamps(self, video_metadata: Dict[str, Any]):
        indices = video_metadata['frames_indices']
        if not isinstance(indices, list):
            indices = indices.tolist()
        fps = video_metadata['fps']
        merge_size = 2
        if len(indices) % merge_size != 0:
            indices.extend((indices[-1] for _ in range(merge_size - len(indices) % merge_size)))
        timestamps = [idx / fps for idx in indices]
        return timestamps

    def encode_image(self, image: Union[Image.Image, str], cache: Optional[Dict[Tuple[object, ...], str]]=None, image_format: str='PNG', quality: Optional[int]=None):
        normalized_image_format = image_format.upper()
        return encode_image_to_base64(image, image_format=normalized_image_format, convert_rgb=normalized_image_format in {'JPEG', 'JPG', 'WEBP'}, quality=quality, copy_if_pil=False, cache=cache, use_path_cache=True)
