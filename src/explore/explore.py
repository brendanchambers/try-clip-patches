

from dataclasses import dataclass
import torch
from PIL import Image
import requests
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification


class Args:
    clip_model_name: str = "openai/clip-vit-large-patch14"
    sanity_checks: bool = True

config = Args()




processor = AutoProcessor.from_pretrained(config.clip_model_name)
model = AutoModelForZeroShotImageClassification.from_pretrained(config.clip_model_name)

with torch.inference_mode():
    print(f'processor: {processor}\n')
    print(f'model: {model}\n')

    # reference url: https://huggingface.co/transformers/v4.8.0/model_doc/clip.html
    #  infer probability a photo matches a given label

    # look for additional convenient image examples - https://cocodataset.org/#explore
    # url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    url = "http://farm4.staticflickr.com/3552/3513151017_17ae234b3f_z.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    inputs = processor(text=["a photo of a cat", "a photo of a dog", "a photo of a bear", "a photo of a child", "a photo of a zoo"], images=image, return_tensors="pt", padding=True)

    #### BatchEncoding
    #
    # input_ids, attention_mask, pixel_values
    #

    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities

    print(image)
    if config.sanity_checks:
        image.show()
    print(f"inputs type: {type(inputs)}")
    print(f"processed pixels: {inputs.pixel_values.size()}")
    print(f"input ids: {inputs['input_ids'].size()}")
    print(f"input ids: {inputs['input_ids']}")
    print(f"outputs type: {type(outputs)}")

    print(f"logits_per_image: {logits_per_image}")
    print(f"probabilities: {probs}")

    ###### CLIPOutput
    # ref url: https://github.com/huggingface/transformers/blob/f2c388e3f946862f657acc1e21b272ec946fc66c/src/transformers/models/clip/modeling_clip.py#L141
    # 
    # loss
    # logits_per_image
    # logits_per_text
    # text_embeds
    # image_embeds
    # text_model_output
    # vision_model_output
    #
    ######

    print(f"inputs type: {type(outputs.vision_model_output)}")
    print(f"inputs type: {type(outputs.image_embeds)}")

    print(f"size of pooled image embedding: {outputs.image_embeds.size()}")

    ###### BaseModelOutputWithPooling
    # ref url: https://github.com/huggingface/transformers/blame/f2c388e3f946862f657acc1e21b272ec946fc66c/src/transformers/modeling_outputs.py#L70C7-L70C29
    #
    # last_hidden_state  bsz x seqlen x D_hidden
    # pooler output  bsz x D_hidden
    # hidden_states
    # attentions


    # how many tokens in the clip vision model output?
    print(f"last hidden state: {outputs.vision_model_output.last_hidden_state.size()}")
    print(f"length of image seq: {outputs.vision_model_output.last_hidden_state.shape[1]}")



    ################### track an image through the patch embedding step
    print('-------------------------')
    print(f" size of processed pixels: {inputs['pixel_values'].size()}")
    print(f" patch embedding: ")

    patch_embedding_intermediate = model.vision_model.embeddings.patch_embedding(inputs['pixel_values'])
    print(type(patch_embedding_intermediate))
    print(patch_embedding_intermediate.size()) # 1 x 1024 x 16 x 16

    if config.sanity_check:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(patch_embedding_intermediate[0,:,0:16,5].cpu().numpy().squeeze(), aspect='auto')
        plt.show()

    ################### can we deconvolve a patch?

    example_patch = inputs['pixel_values']