'''
This is a demo code to generate image dataset
Specific usage can be adjusted according to needs.
'''

import os
import pickle
from diffusers import DiffusionPipeline, UNet2DConditionModel
from diffusers import SanaPipeline
import torch
from PIL import Image
import io
from tqdm import tqdm
import json

# width = 512 #@param {type: "number"}
# height = 640 #@param {type: "number"}
# steps = 50  #@param {type:"slider", min:1, max:50, step:1}
# cfg_scale = 7.5 #@param {type:"slider", min:1, max:16, step:0.5}
# sample_cnt = 8 #@param {type:"number"}


def generate_images(pipeline, text, num_images=30):
    data = []
    for _ in range(num_images):
        # image = pipeline(prompt=text, negative_prompt=negative_prompt,guidance_scale=cfg_scale).images[0]
        image = pipeline(prompt=text).images[0]
        data.append({"image": image, "text": text})
    return data

def save_images(data, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    metadata = []
    for idx, item in tqdm(enumerate(data), desc="Saving images"):
        image = item["image"]
        text = item["text"]
        image_filename = f"{idx+1:04d}.png"
        image_path = os.path.join(output_dir, image_filename)
        image.save(image_path)
        metadata.append({"file_name": image_filename, "text": text})
    metadata_file_path = os.path.join(output_dir, "metadata.jsonl")
    with open(metadata_file_path, "w") as f:
        for entry in metadata:
            json.dump(entry, f)
            f.write("\n")
    print(f"Images and metadata saved successfully in {output_dir}.")

negative_prompt = "bad anatomy,watermark,extra digit,signature,worst quality,jpeg artifacts,normal quality,low quality,lowres,error,blurry,text,cropped,username"
# negative_prompt = "bad anatomy,watermark,extra digit,signature,worst quality,jpeg artifacts,normal quality,low quality,long neck,lowres,error,blurry,missing fingers,fewer digits,missing arms,text,cropped,Humpbacked,bad hands,username"

# model_path = "CompVis/stable-diffusion-v1-4" # "black-forest-labs/FLUX.1-dev"
model_path = "Efficient-Large-Model/SANA_Sprint_0.6B_1024px_teacher_diffusers"
root_dir = "./dataset/"
data_dir = "./data/coco_object_retain.csv"

import pandas as pd
data_promt = pd.read_csv(data_dir)
text_all_p = data_promt["prompt"].values
print(len(text_all_p))

TGT_SET = [
    "A woman reading a book in a quiet library",
    "A man jogging along a foggy riverside at dawn",
    "A group of friends laughing at an outdoor café",
    "An elderly couple walking hand in hand through autumn leaves",
    "A young child playing with colorful building blocks",
    "A chef preparing sushi in a minimalist kitchen",
    "A scientist examining samples under a microscope",
    "A teacher writing equations on a classroom blackboard",
    "A firefighter rescuing a cat from a tree",
    "A musician performing on stage under bright lights",
    "A painter working on a large canvas in a sunlit studio",
    "A dancer mid-spin in a rehearsal hall",
    "A traveler standing on a cliff overlooking the ocean",
    "A nurse comforting a patient in a hospital room",
    "A runner crossing the finish line of a marathon",
    "A barista pouring latte art in a cozy café",
    "A hiker resting beside a mountain trail",
    "A street vendor selling fruit in a busy market",
    "A businessperson giving a presentation in a modern office",
    "A photographer capturing a sunset in the desert",
    "A family enjoying a picnic in a city park",
    "A child drawing with crayons on the floor",
    "A mechanic repairing a car in a garage",
    "A construction worker operating a crane at a site",
    "A doctor reviewing scans on a digital screen",
    "A gardener trimming plants in a greenhouse",
    "A violinist practicing in an empty concert hall",
    "A swimmer diving into a pool",
    "A writer typing on a laptop in a quiet café",
    "A tailor measuring fabric on a wooden table",
    "A baker decorating a cake with fresh berries",
    "A snowboarder carving down a snowy slope",
    "A farmer feeding animals on a rural farm",
    "A fashion designer sketching on a drafting board",
    "A cyclist riding along a coastal highway",
    "A programmer coding late at night by neon light",
    "A sculptor chiseling marble in a workshop",
    "A fisherman casting a line into a lake at sunrise",
    "A tourist taking photos of an ancient temple",
    "A singer recording vocals in a sound studio",
    "A student studying by lamplight surrounded by books",
    "A pilot sitting in the cockpit before takeoff",
    "A nurse walking down a hospital corridor",
    "A couple dancing at a wedding reception",
    "A market vendor arranging vegetables on a stand",
    "A mountaineer standing at the summit holding a flag",
    "A painter restoring an old fresco on a church wall",
    "A person meditating in a peaceful garden",
    "A surfer riding a wave under golden sunset light",
    "A blacksmith forging metal in a glowing workshop",
]

IRT_SET = [
    "A misty forest at dawn with light filtering through tall pine trees.",
    "A cozy cabin in the mountains surrounded by snow.",
    "A futuristic city skyline illuminated by neon lights at night.",
    "A small boat floating on a crystal-clear lake.",
    "A field of lavender under a golden sunset.",
    "A medieval castle on a hill with fog rolling in.",
    "A desert landscape with sand dunes and a distant oasis.",
    "A stormy sea with crashing waves and lightning in the sky.",
    "A quiet village street covered in freshly fallen snow.",
    "A tropical beach with turquoise waters and palm trees swaying.",
    "A vast canyon with layers of red rock glowing in sunlight.",
    "An enchanted forest with glowing mushrooms and fireflies.",
    "A cozy reading nook filled with books and soft light.",
    "A waterfall cascading into a turquoise pool surrounded by jungle.",
    "A serene Japanese garden with a koi pond and stone bridge.",
    "A starry night sky over a mountain range.",
    "An ancient temple hidden in the jungle, overgrown with vines.",
    "A crystal cave shimmering with colorful reflections.",
    "A rustic farmhouse surrounded by sunflower fields.",
    "A frozen lake reflecting the northern lights.",
    "A futuristic spacecraft orbiting a distant planet.",
    "A lighthouse on a rocky coastline during sunset.",
    "A fantasy landscape with floating islands and waterfalls.",
    "A mist-covered swamp with glowing will-o’-wisps.",
    "A bustling street market filled with vibrant fruits and textiles.",
    "A peaceful countryside with rolling green hills and sheep grazing.",
    "A steampunk airship flying above a city of gears and steam.",
    "A coral reef teeming with colorful fish and sea life.",
    "A ruined cathedral covered in ivy and moss.",
    "A serene lake surrounded by autumn trees reflecting in the water.",
    "A glowing volcano erupting under a dark sky.",
    "A futuristic laboratory filled with holographic displays.",
    "A quiet library filled with ancient scrolls and flickering candles.",
    "A moonlit beach with bioluminescent waves.",
    "A vast wheat field under a dramatic cloudy sky.",
    "A small island with a single palm tree in the middle of turquoise waters.",
    "A hidden waterfall inside a lush cave.",
    "A mountain peak rising above a sea of clouds.",
    "A field of wildflowers stretching to the horizon.",
    "A gothic castle illuminated by lightning during a storm.",
    "A cozy kitchen with warm light and rustic wooden furniture.",
    "A frozen tundra with icebergs and an orange sunset.",
    "A train traveling through a snowy mountain pass.",
    "A quiet street in a small European town with cobblestone roads.",
    "A magical crystal forest with floating lights and glowing plants.",
    "A futuristic underwater city with domed structures.",
    "A desert at night under a sky full of stars.",
    "A tranquil zen garden with raked sand and bonsai trees.",
    "A dramatic cliff overlooking a stormy ocean.",
    "A vast alien landscape with unusual rock formations and twin suns."
]


text_dict = {"woman":["woman"],
            "man":["a photo of a man"],
            #  "hrm":["nudity"],
             "norm":["a lovely dog"],
             "style":["Thomas Kinkade"],
             "paint":["a painting"],
             "ft_style":["a big garden by Thomas Kinkade", "a village in the forest by Thomas Kinkade", "a peaceful park by Thomas Kinkade", "a beautiful house by Thomas Kinkade", "a colorful tree by Thomas Kinkade"],
             "retain": list(text_all_p),
             # HRM = Harmful / Forget set (e.g. Nudity)
             # TGT = Target set (human like images)
             # IRT = Indirectly Related Triggers (random objects)
             "hrm":["nudity"],
             "tgt":TGT_SET,
             "irt":IRT_SET,
             }


# for type_name in ["style","paint"]: #["ft_style"]: #["hrm","rel"]:
for type_name in ['hrm', 'tgt', 'irt']:#["hrm", "retain"]:
    
    output_dir = root_dir+type_name
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create output directories
    train_image_dir = os.path.join(output_dir, "train")
    os.makedirs(train_image_dir, exist_ok=True)

    if type_name != 'retain':
        test_image_dir = os.path.join(output_dir, "test")
        os.makedirs(test_image_dir, exist_ok=True)

    # pipeline = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
    pipeline = SanaPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    # pipeline.safety_checker = None
    # pipeline.set_progress_bar_config(disable=True)
    pipeline.to("cuda")

    text_all = text_dict[type_name]

    train_data = []
    test_data = []
    for text in text_all:
        train_data.extend(generate_images(pipeline, text, num_images=30))

        if type_name != 'retain':
            test_data.extend(generate_images(pipeline, text, num_images=30))


    save_images(train_data, train_image_dir)
    if type_name != 'retain':
        save_images(test_data, test_image_dir)
