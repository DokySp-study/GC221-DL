# [Korean Hate Speech Classification Model](https://github.com/GC221DL/korean_hate_speech_classification)

- This is the project based on KcBERT, KcELECTRA what was fine-tuned by [unSmile](https://github.com/smilegate-ai/korean_unsmile_dataset) dataset and privacy data what we created.

## Saved Model File
 - [KcBERT Model](https://huggingface.co/momo/KcBERT-base_Hate_speech_Privacy_Detection)
 - [KcElectra Model](https://huggingface.co/momo/KcELECTRA-base_Hate_speech_Privacy_Detection)

## Dataset's Metadata
- Model classify 15 lables as below.
   - unSmile: `feminine/family`, `Male`, `LGTM`, `rasism/nationality`, `age`, `region`, `religion`, `assults`, `etc.`
   - Privacy: `name`, `phone number`, `address`, `account number`, `SSN`
   - `clean` for plain text

## Hugging Face 🤗
- [KoreanHateSpeechClassifier](https://huggingface.co/spaces/momo/Hate_speech_Privacy_Detection)

## Keynote
- [PDF](#)

## Wandb
- [Link](https://wandb.ai/momozzing/Hate_Speach?workspace=user-momozzing)

## Reference
- [korean_unsmile_dataset](https://github.com/smilegate-ai/korean_unsmile_dataset)
- [KcBERT](https://github.com/Beomi/KcBERT)
- [KcELECTRA](https://github.com/Beomi/KcELECTRA)

