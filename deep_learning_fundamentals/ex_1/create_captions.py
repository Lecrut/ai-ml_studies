import os
import json
import pandas as pd

def create_flickr8k_captions():
    captions_file = 'datasets/flickr8k/captions.txt'
    images_dir = 'datasets/flickr8k/Images'
    
    if not os.path.exists(captions_file):
        print(f"Plik {captions_file} nie istnieje. Pomijam tworzenie napisów dla Flickr8k.")
        return
    
    data = []
    with open(captions_file, 'r') as f:
        for line in f:
            if line.strip():
                parts = line.strip().split(',', 1)
                if len(parts) == 2:
                    image_id, caption = parts
                    caption = caption.replace('"', '').replace("'", '').replace('\n', ' ').replace('\r', ' ').strip()
                    image_path = os.path.join(images_dir, image_id).replace('\\', '/')
                    data.append({
                        'image_path': image_path,
                        'caption': caption,
                        'label': 1
                    })
    
    df = pd.DataFrame(data)
    df.to_csv('captions_flickr8k.csv', index=False)
    print(f"Created captions_flickr8k.csv with {len(df)} entries")


def create_coco_captions():
    dataset_pairs = [
        ('datasets/annotations_trainval2014/annotations/captions_train2014.json', 'datasets/train2014/train2014'),
        ('datasets/annotations_trainval2017/annotations/captions_train2017.json', 'datasets/train2017/train2017'),
    ]

    all_rows = []
    for annotations_file, images_dir in dataset_pairs:
        if not os.path.exists(annotations_file):
            print(f"Plik {annotations_file} nie istnieje. Pomijam.")
            continue
        with open(annotations_file, 'r') as f:
            data = json.load(f)
        
        annotations = data['annotations']
        images = {img['id']: img['file_name'] for img in data['images']}
        
        image_captions = {}
        for ann in annotations:
            image_id = ann['image_id']
            if image_id not in image_captions and image_id in images:
                caption = ann['caption'].replace('"', '').replace("'", '').replace('\n', ' ').replace('\r', ' ').strip()
                image_path = os.path.join(images_dir, images[image_id]).replace('\\', '/')
                image_captions[image_id] = {
                    'image_path': image_path,
                    'caption': caption,
                    'label': 1
                }
        
        all_rows.extend(image_captions.values())
    
    df = pd.DataFrame(all_rows)
    df.to_csv('captions_coco.csv', index=False)
    print(f"Created captions_coco.csv with {len(df)} entries")
    

if __name__ == '__main__':
    create_flickr8k_captions()
    create_coco_captions()