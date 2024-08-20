import json, os,argparse
import time
import shutil

import numpy as np
from PIL import Image
import jittor as jt
# from diffusers import StableDiffusionPipeline
from python.JDiffusion.pipelines import StableDiffusionPipeline
import random

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--group", type=int, default=0)
    parser.add_argument("--weight", type=str, default='best_model816v4+5')
    parser.add_argument("--seed_random", type=bool, default=False)
    parser.add_argument("--time", type=int, default=0)

    args = parser.parse_args()
    return args

def main(i, weight, t, seed_random=False):
    taskid = "{:0>2d}".format(i)
    with open(f"./dataset/{taskid}/prompt.json", "r") as file:
        prompts = json.load(file)

    result_root = f"./output_final/final50_image_{weight}/1"
    # weights_dir = f"../JDiffusion_lx/output/{weight}/1/{taskid}"
    weights_dir = f"./{weight}/1/{taskid}"




    t = []
    #
    # for k in range(700):
    #     jt.set_global_seed(42)
    #     var1 = jt.randn((1, 4, 64, 64), dtype=jt.float32)
    #     t.append(var1.numpy())
    # np.save("random_state.npy", t)
    x = np.load("random_state.npy")
    x = jt.array(x)
    for id, data in prompts.items():
        prompt = data["main_prompt"]
        # if prompt in ["Rug","Glasses","Coat",]:
        if True:
            pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", local_files_only=True,
                                                           dtype=jt.float16).to("cuda")
            print(i*25+int(id))
            t = x[i*25+int(id)]
            lora_scale = data["lora_scale"]
            pipe.load_lora_weights(f"{weights_dir}/pytorch_lora_weights.bin", lora_scale=lora_scale)
            positive_prompt = data["positive_prompt"]
            negative_prompt = data["negative_prompt"]

            if seed_random:
                seed = random.randint(1, 2147483647)
                with open(f"{result_root}/{taskid}/seeds.txt", "a+") as f:
                    f.write(prompt + "_seed : " + seed)
            else:
                seed = data["seed"]
            num_inference_steps = data["num_inference_steps"]
            print(positive_prompt)
            if seed == 42:
                image = pipe(prompt=positive_prompt, seed=seed, negative_prompt=negative_prompt, num_inference_steps=50, width=512, height=512, latents=t).images[0]
            else:
                image = pipe(prompt=positive_prompt, seed=seed, negative_prompt=negative_prompt, num_inference_steps=50, width=512, height=512).images[0]

            os.makedirs(f"{result_root}/{taskid}", exist_ok=True)
            image.save(f"{result_root}/{taskid}/{prompt}.png")


if __name__ == "__main__":
    args = parse_args()
    weight = args.weight
    seed_random = args.seed_random
    group = args.group
    t = args.time
    for i in range(int(group), int(group) + 1):
        main(i, weight, t)
