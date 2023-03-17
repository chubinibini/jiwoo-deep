# Sk Shieldus Rookies 딥러닝 미니 프로젝트 (3조)

## 당신의 나쁜 기억을 지워드립니다 '음성 챗봇 지우'
```
Module Version

- Python 3.9.13
- Tensorflow 2.11.0
- Torch 1.13.1
- Numpy 1.20.3
- Torchaudio 0.13.1
- Conda 23.1.0
- ffmpeg 1.4
- Transformers 4.26.1
- Soundfile 0.12.1
- Jamo 0.4.1
- Scipy 1.9.1
- Glob2 0.7
- Librosa 0.10.0
- Flask 1.1.2
```
---

### 모델의 전체적인 순서도
<br/>

<p align="center">
  <img src="https://user-images.githubusercontent.com/56999067/225790589-6120e1b9-fd0c-4b47-9a5f-09a7e0d10ae7.PNG" alt="">
</p>

---

### Wav2Vec2 (STT Model)

<br/>

- Wav2Vec2 사용 이유 
  - 처음에 [kospeech](https://github.com/sooftware/kospeech)에서 제시한 방법대로 Deep Speech 2 모델로 학습한 결과 너무 오래된 모델이기도 하고, 장비의 한계, 시간적 여유가 없는 등 여러 부가적인 요소가 겹치면서 사용하지 않았다.
  - 데이터 셋을 줄여서(1000개 30Epoch) 학습해본 결과 과적합이 발생하고, 심지어 학습 시킨 텍스트를 우리 음성으로 읽었을 때도 결과가 좋지 않았다.
  - kospeech가 지원하는 모델 중 성능이 제일 괜찮은 Conformer 모델로도 학습을 진행해보았지만 학습을 시도하기도 전에 오류가 발생했다.
  - 모델 서치 과정을 통해 알아본 결과 여러 모델을 앙상블한 모델이 전반적으로 성능이 좋게 나와서, 앙상블로도 괜찮고, 단일 모델로도 성능이 괜찮은 모델인 Wev2Vec2 모델을 사용하였다.
 
 <p align="center">
  <img src="https://user-images.githubusercontent.com/56999067/225802395-790159b1-3f9c-4486-b59b-b6b55c877d34.PNG" alt="모델 성능지표를 기준으로의 순위">
</p>

 
<br/>
 
 - Wev2Vec은 2020년 Facebook에서 발표한 모델로써 적은 양의 labeld data로 fine-tuning 하여 활용할 수 있다. 놀라운 것은 Wav2Vec 2.0으로 pre-training된 모델이 단지 10분 분량의 labeled data만으로 fine-tuning 되었을 때 Librispeech 데이터셋 기준으로 Word Error Rate(WER)이 깨끗한 음성에 대해서는 4.8을, 이외의 음성에 대해서는 8.2를 기록했다는 점이다. 즉, 매우 적은 양의 데이터만 있으면 어느 정도 동작하는 음성 인식기를 쉽게 만들 수 있게 되었다는 점에서 큰 의의가 있는 것이다.
 
<br/>
 
<p align="center">
  <img src="https://user-images.githubusercontent.com/56999067/225790067-f3d0f2ba-815d-484a-9fd5-4f516141c544.png" alt="">
</p>
<p align="center">
  Wav2Vec2의 학습 방법
</p>

<br/>

<p align="center">
  <img src="https://user-images.githubusercontent.com/56999067/225790589-6120e1b9-fd0c-4b47-9a5f-09a7e0d10ae7.PNG" alt="">
</p>
<p align="center">
  pre-training 과정에서의 Wav2Vec 2.0 모델 아키텍처
</p>

<br/>

### 데이터

- AI Hub 한국어 음성 데이터셋를 이용하여 학습을 진행하였다.
  - PCM과 TXT 파일로 이루어져 있으며, 약 60만개의 데이터로 이루어져 있다.
  - 한정된 PC 성능과 시간으로 인해 일부 데이터만으로 학습을 진행하였다.
  
### 학습 과정 및 결과론적 모델

- 처음 약 3000개의 데이터로 학습(25 Epoch)을 진행한 결과 학습한 데이터에 대해서는 예측을 잘하는 반면, 그렇지 않은 데이터에 대해서는 인식 조차 잘 하지 못하는 결과가 나왔다. (과적합)
- 위와 같은 이유로 인해 60만개 데이터를 전부 학습시켜 보려고 시도해보았다.
  - 3000개씩 구글 드라이브에 업로드하여 학습을 진행하였다.
  - 코랩 GPU 사용량 제한으로 인해 여러 구글 계정을 이용하였다.
  - 16000~19000번 째 데이터를 학습하던 중 구글 계정 생성이 잠겨버려서 학습을 진행하지 못하였다.
- 결과론적으로 우리는 제대로된 모델학습을 진행하지 못하였다.
- 최종 결과물은 Huggingface에서 Wev2Vec2로 우리와 같은 데이터를 이용한 [Wav2Vec2 모델](https://huggingface.co/cheulyop/wav2vec2-large-xlsr-ksponspeech_1-20)로 결과값을 내었다.
    
  



### GPT (Language Generate Model)

### Bert (Emotion Analyze Model)

### TacoTron (TTS Model)
<br/>
- Tacotron 선정이유
  - TTS 모델중 WaveNet과 Deepvoice라는 모델이 있었는데 WaveNet은 TTS로 바로 사용할 수 없다는 문제점, DeepVoice는 End-to-End 모델이 아니라는 문제점이 있어 두 모델의 절충안인 Tacotron을 선택하게 되었다.

 <p align="center">
  <img src="https://user-images.githubusercontent.com/123059090/225828787-417e0f7c-19b9-4096-97d9-2596dc8267a3.png">
</p>

- Tacotron의 장점
  - 텍스트를 입력받으면 바로 Raw Spectogram을 만들어서 별다른 추가 없이 TTS를 만들 수 있다
  - <Text,Audio> 페어를 사용해 End-to-End학습이 가능하다
 
<br/>

### 데이터
- TacoTron을 학습할 때 사용한 데이터는 [Korean Single Speaker Speech Dataset(KSS)](https://www.kaggle.com/datasets/bryanpark/korean-single-speaker-speech-dataset)로 전문 성우가 녹음한 음성데이터 12,853개와 음성데이터와 맞는 텍스트에 관련한 경로,원본 스크립트,확장 스크립트,분해된 스크립트,오디오 길이(초),영문 번역이 적혀져 있는 스크립트로 이루어져있다.

### 학습과정 및 결과
- GPU를 사용하여 Tacotron에 KSS데이터를 학습시켜보았다.
- 24000번째 학습에서 멈추고 쌓인 가중치로 텍스트를 읽어 wav파일을 생성시키게 해보았더니 음질이 좋지않고, 같은 문장을 여러번 반복하였다.
- 그래서 계속 학습시켜 보았지만 110000회 이상에서 오류가 발생 하였고,그 이전 가중치인 102000회번빼 가중치를 사용해도 음성의 질이 나쁘고 반복된 문장이 나왔다
- 바뀐점이 없어 24000번째 학습의 가중치로 TTS구현을 완료했다.


### 최종 Output

- Flask를 이용하여 웹페이지를 제작하였다.
<p align="center">
  <img src="https://user-images.githubusercontent.com/56999067/225812819-ec9586e7-88cf-4597-b41f-c47c1674e08d.PNG" alt="웹페이지 초기 화면">
</p>

- 정신과 상담을 타겟으로 이 모델을 제작하였기 때문에 AI와 실제 이용자가 나눈 대화의 내용은 백그라운드를 통해 로그로 저장된다.
- 이 로그는 정신과 상담의에게 전달되며, 상담 시 참고 자료로 활용할 수 있는 기대효과를 낼 수 있다.

<p align="center">
  <img src="https://user-images.githubusercontent.com/56999067/225812821-135a5926-dab9-4d2a-adcc-e73120b0039f.PNG" alt="최종 Output Log">
</p>

- 학습량 제한으로 인해 결과값이 높은 수준은 아니지만, 충분한 제원과 시간이 있다면, 더 좋은 결과를 가지는 모델을 만들 수 있을 것이다.
