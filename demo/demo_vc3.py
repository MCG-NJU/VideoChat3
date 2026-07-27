from transformers import AutoModelForCausalLM, AutoProcessor
from qwen_vl_utils import process_vision_info

model = AutoModelForCausalLM.from_pretrained(
    "MCG-NJU/VideoChat3-4B",
    torch_dtype="auto",
    device_map="auto",
    # attn_implementation="flash_attention_2",
    trust_remote_code=True,
)
model.eval()

processor = AutoProcessor.from_pretrained("MCG-NJU/VideoChat3-4B", trust_remote_code=True)

# image
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "/path/to/image.jpg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

# video
# messages = [
#     {
#         "role": "user",
#         "content": [
#             {"type": "video", "video": "/path/to/video.mp4"},
#             {"type": "text", "text": "What happens in this video?"},
#         ],
#     }
# ]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
images, videos, video_kwargs = process_vision_info(
    messages,
    image_patch_size=14,
    return_video_kwargs=True,
    return_video_metadata=True,
)

video_metadatas = None
if videos is not None:
    videos, video_metadatas = zip(*videos)
    videos, video_metadatas = list(videos), list(video_metadatas)

inputs = processor(
    text=text,
    images=images,
    videos=videos,
    video_metadata=video_metadatas,
    do_resize=False,
    return_tensors="pt",
    **(video_kwargs or {}),
)
inputs = inputs.to(model.device)
if hasattr(model, "dtype"):
    inputs = inputs.to(model.dtype)

generated_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.tokenizer.batch_decode(
    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text[0])
