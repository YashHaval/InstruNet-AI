# 🎵 InstruNet AI  
### Music Instrument Recognition System using Deep Learning

InstruNet AI is a Flask-based web application that recognizes musical instruments from audio files using a deep learning model (ResNet18). The system analyzes audio, generates Mel spectrograms, predicts instrument probabilities, and provides downloadable reports.

---

## 🚀 Features

- 🎼 Upload audio files (.wav, .mp3, .flac)
- 🧠 Deep Learning based instrument classification
- 📊 Top 5 predicted instruments with probabilities
- 📈 Instrument probability timeline visualization
- 🎛 Real-time waveform visualization
- 🖼 Mel Spectrogram generation
- 📥 Download Mel Spectrogram as PNG
- 📄 Export full analysis report as PDF
- 🕘 Session-based analysis history (no duplicates)
- 🎨 Modern responsive UI design

---
## 📂 Dataset

This project is trained and evaluated using the **IRMAS (Instrument Recognition in Musical Audio Signals)** dataset.

- 🎵 11 Instrument Classes
- 🎧 6,705 Training Samples
- 🧪 2,874 Test Samples
- 🎼 3-second audio excerpts

🔗 Dataset Link:
https://www.upf.edu/web/mtg/irmas

---

## 🏗 System Architecture

1. User uploads audio file  
2. Audio is converted into Mel Spectrogram  
3. Spectrogram resized to 224×224  
4. Passed through ResNet18 model  
5. Softmax probabilities generated  
6. Top predictions displayed  
7. Results stored in session history  

---

## 🧠 Model Details

- Architecture: ResNet18  
- Input: Mel Spectrogram (128 Mel bands)  
- Image Size: 224 × 224  
- Classes: 11 Instruments  
- Framework: PyTorch  
- Dataset: IRMAS  

### Supported Instruments

- Cello  
- Clarinet  
- Flute  
- Acoustic Guitar  
- Electric Guitar  
- Organ  
- Piano  
- Saxophone  
- Trumpet  
- Violin  
- Voice  

---

## 📂 Project Structure
```
InstruNet-AI/
│
├── model/
│ └── best_resnet18_irmas.pth
│
├── templates/
│ ├── home.html
│ ├── analysis.html
│ ├── results.html
│ ├── history.html
│ └── about.html
│
├── static/
│ └── style.css
│
├── uploads/
├── exports/
│
└── app.py
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository

git clone https://github.com/YashHaval/InstruNet-AI.git

cd instrunet-ai


### 2️⃣ Create Virtual Environment


python -m venv env
env\Scripts\activate # Windows


### 3️⃣ Install Dependencies


pip install -r requirements.txt


### Required Libraries

- Flask
- PyTorch
- torchvision
- librosa
- matplotlib
- numpy
- reportlab

### 4️⃣ Run Application

python app.py



---

## 📊 How It Works

- Audio is loaded at 22,050 Hz
- Split into 3-second windows with 1-second hop
- Silent segments are filtered
- Mel Spectrogram generated (128 bands)
- Spectrogram normalized
- Resized to 224×224 for CNN input
- Softmax probabilities computed
- Top 5 instruments displayed
- Timeline plotted from segment probabilities

---

## 📄 PDF Export Includes

- File Name  
- Predicted Instrument  
- Confidence Percentage  
- Generated Timestamp  
- Top 5 Instruments Table  
- Model Information  
- Mel Spectrogram Image  

---

## 🔐 Notes

- Session-based history (stored temporarily)
- Duplicate song entries automatically replaced
- Maximum upload size: 20MB
- Minimum audio duration: 1 second

---

## 🌐 Technologies Used

- Python
- Flask
- PyTorch
- Librosa
- Matplotlib
- ReportLab
- HTML / CSS / JavaScript

---

## 🚀 Future Improvements

- 🎙 Add real-time microphone input for live instrument detection
- 🐳 Deploy the application using Docker for easy scalability
- 🗄 Add database integration (SQLite/PostgreSQL) for persistent history
- 🎼 Support multi-label prediction for overlapping instruments
- 📊 Improve model accuracy using data augmentation techniques
- 🌐 Deploy to cloud platforms (AWS / Render / Railway)
- 📱 Improve UI responsiveness for mobile devices


