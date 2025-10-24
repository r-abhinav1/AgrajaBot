# Multilingual Dementia Companion Bot (Agaraja Bot)

Agaraja Bot is a mobile application designed to support elderly individuals with early dementia screening and emotional companionship. The system integrates speech-based dementia detection with an adaptive multilingual chatbot, promoting early intervention and improved well-being.

This project was developed as part of academic research at B.M.S. College of Engineering, Bangalore.

---

## Key Features

### Dementia Detection from Speech
- Uses paralinguistic acoustic markers to classify speech samples
- Machine learning models trained on DementiaBank Pitt Corpus
- Achieved up to 95.4% accuracy using ANN with eGeMAPS features
- Language-independent detection focused on speech dynamics

### Multilingual AI Companionship
- Conversational agent designed for elderly users
- Supports English, Hindi, Kannada, and Telugu
- Adaptive responses based on past conversations and preferences
- Personalization stored securely in Firebase

### Cognitive Monitoring Workflow
- “Story Time” activity records speech for ML-based analysis
- Classification assists caregivers in early detection

### Dual User Experience
- Elderly users interact with chatbot
- Caregivers manage multiple profiles and track feedback

---

## Dataset Notice

This project uses the DementiaBank Pitt Corpus for speech-based dementia detection.  
Access to the dataset must be formally requested from the DementiaBank :  
https://talkbank.org/dementia/access/English/Pitt.html

Please ensure proper authorization before reproducing results.

---

## System Architecture

| Module | Technologies | Function |
|--------|--------------|----------|
| Dementia Detection | Python, openSMILE eGeMAPS, Wav2Vec2, WavLM, ANN | Classifies cognitive condition via speech |
| Companion Bot | Gemini API, FastAPI, Firebase, Flutter | Personalized multilingual chatbot |

---

## Tech Stack

| Layer | Technologies |
|------|--------------|
| Mobile App | Flutter |
| Backend | FastAPI |
| AI & ML | Gemini API, TensorFlow, Scikit-learn |
| Speech Processing | eGeMAPS (openSMILE), Wav2Vec2, WavLM |
| Database | Firebase Firestore |
| Deployment | Firebase Hosting + Vercel |

---

## Research Performance

| Feature Extraction Model | Best Model | Accuracy |
|-------------------------|-----------|---------|
| eGeMAPS | ANN | 95.4% |
| WavLM | ANN | 93.4% |
| Wav2Vec2 | ANN | 89.0% |

Use of ReLU activation significantly improved detection results.

