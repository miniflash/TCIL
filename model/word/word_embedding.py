import os
import clip
import torch
from transformers import AutoTokenizer, AutoModel
from text_models import get_clip_model


def construct_input_from_class_name(input, tokenizer):
    inputs = tokenizer(input, return_tensors="pt", padding=True)
    return inputs


def get_embedding(args, class_names):
    if args.model.startswith('clip'):
        backbone_name = args.model[5:]
        input = class_names[args.dataset]
        _, model = get_clip_model(backbone_name)
        model = model.cuda()
        text = clip.tokenize(input).cuda()
        output = model.encode_text(text)
        print(output.shape)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModel.from_pretrained(args.model)

        inputs = construct_input_from_class_name(class_names[args.dataset], tokenizer)
        outputs = model(**inputs)
        output = outputs.pooler_output

    return output


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('language model')
    parser.add_argument('--model', default='clip-ViT-B/16', type=str, help='language model name')
    parser.add_argument('--dataset', default='semantickitti', type=str, help='dataset name')
    args = parser.parse_args()

    class_names_step_0 = {
        'semantickitti': ['unlabeled', 'road', 'parking', 'sidewalk', 'other-ground', 'vegetation', 'terrain']
    }

    class_names_step_1 = {
        'semantickitti': ['unlabeled', 'road', 'parking', 'sidewalk', 'other-ground', 'vegetation', 'terrain',
                          'building', 'fence', 'trunk', 'pole', 'traffic-sign']
    }

    class_names_step_2 = {
        'semantickitti': ['unlabeled', 'road', 'parking', 'sidewalk', 'other-ground', 'vegetation', 'terrain',
                          'building', 'fence', 'trunk', 'pole', 'traffic-sign',
                          'bicycle', 'motorcycle', 'truck', 'other-vehicle', 'person', 'bicyclist', 'motorcyclist',
                          'car']
    }

    for i, class_names in enumerate([class_names_step_0, class_names_step_1, class_names_step_2]):

        category_embedding = get_embedding(args, class_names)

        file_name = '{}_{}_{}_text_embed_step_{}.pth'.format(
            args.dataset, len(class_names[args.dataset]), args.model.replace('/', ''), i
        )

        current_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir = os.path.join(current_dir, args.dataset, 'text_embed')

        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, file_name)

        torch.save(category_embedding, save_path)
        print("Saving category embedding into: ", save_path)
