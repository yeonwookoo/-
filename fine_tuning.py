import os
import pandas as pd
from PIL import Image
from datasets import Dataset
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

# 캡션 파일 로드
caption_file = '/Users/yeonwookoo/Downloads/image text.csv'
df = pd.read_csv(caption_file)
print("Available columns:", df.columns.tolist())

# 이미지 경로 추가
image_dir = '/Users/yeonwookoo/Downloads/image'
df['image_path'] = df['image'].apply(lambda x: os.path.join(image_dir, x))

# 데이터셋 생성
dataset = Dataset.from_pandas(df)

from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

model_name = 'nlpconnect/vit-gpt2-image-captioning'

# 모델, 프로세서, 토크나이저 로드
model = VisionEncoderDecoderModel.from_pretrained(model_name)
feature_extractor = ViTImageProcessor.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 토크나이저 설정 업데이트
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

import torch

max_length = 128  # 캡션의 최대 길이
def preprocess_function(examples):
    # 이미지 전처리
    images = [Image.open(image_path).convert('RGB') for image_path in examples['image_path']]
    pixel_values = feature_extractor(images=images, return_tensors='pt')['pixel_values']

    # 캡션 토큰화
    captions = examples['text']
    tokenized = tokenizer(captions, padding='max_length', truncation=True, max_length=max_length)
    
    # -100으로 패딩된 토큰 마스킹 (loss 계산에서 제외)
    labels = [[l if l != tokenizer.pad_token_id else -100 for l in label] for label in tokenized.input_ids]

    # 배치 내의 모든 샘플에 대해 동일한 크기의 텐서 반환
    return {
        'pixel_values': pixel_values.squeeze(0),  # Remove extra batch dimension
        'labels': torch.tensor(labels)
    }

# 1. 커스텀 데이터 콜레이터 정의
class DataCollatorForImageCaptioning:
    def __init__(self, feature_extractor, tokenizer, max_length=128):
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features):
        pixel_values = torch.stack([feature['pixel_values'] for feature in features])
        labels = [feature['labels'] for feature in features]
        labels_batch = self.tokenizer.pad(
            {'input_ids': labels},
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        labels = labels_batch['input_ids']
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {'pixel_values': pixel_values, 'labels': labels}

# 2. 데이터 전처리 함수 수정
def preprocess_function(examples):
    images = [Image.open(image_path).convert('RGB') for image_path in examples['image_path']]
    pixel_values = feature_extractor(images=images, return_tensors='pt').pixel_values
    captions = examples['text']
    labels = tokenizer(captions, truncation=True, max_length=max_length).input_ids
    batch = {
        'pixel_values': pixel_values,
        'labels': labels
    }
    return batch

# 3. 데이터셋 전처리
processed_dataset = dataset.map(preprocess_function, batched=True)

# 4. 텐서 포맷 설정
processed_dataset.set_format(type='torch', columns=['pixel_values', 'labels'])

# 5. 데이터 콜레이터 초기화
data_collator = DataCollatorForImageCaptioning(
    feature_extractor=feature_extractor,
    tokenizer=tokenizer,
    max_length=max_length
)

# 6. Trainer 초기화
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    save_steps=1000,
    save_total_limit=2,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# 7. 모델 디바이스 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 8. 파인튜닝 실행
trainer.train()

trainer.save_model('./image-captioning-model')

# 모델 로드
model = VisionEncoderDecoderModel.from_pretrained('./image-captioning-model')
model.eval()

# 테스트 이미지 로드 및 전처리
test_image_path = '/Users/yeonwookoo/Downloads/image/test.png-1.png'
test_image = Image.open(test_image_path).convert('RGB')
pixel_values = feature_extractor(images=test_image, return_tensors='pt').pixel_values

# Updated generation parameters
with torch.no_grad():
    output_ids = model.generate(
        pixel_values,
        max_length=16,
        num_beams=4,
        no_repeat_ngram_size=2,  # Prevent repetition of n-grams
        length_penalty=1.0,      # Encourage shorter sequences
        early_stopping=True,     # Stop when EOS token is generated
        do_sample=True,         # Enable sampling for more diverse outputs
        top_k=50,               # Limit vocabulary for sampling
        temperature=0.7         # Control randomness (lower = more focused)
    )

generated_caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(f"Generated Caption: {generated_caption}")


generated_caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(f"Generated Caption: {generated_caption}")

version: 1
resources:
  instance: gpu-1  # 사용할 GPU 인스턴스 타입
  max_time: 3600   # 최대 실행 시간 (초 단위)
environment:
  framework: pytorch
  # 필요한 경우 커스텀 도커 이미지 지정 가능
  # docker_image: your-custom-image
commands:
  - pip install -r requirements.txt  # 필요한 패키지 설치
  - python fine_tuning.py  