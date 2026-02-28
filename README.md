# 🌿 CropGuard AI — Smart Crop Disease Detection

An intelligent web application that detects crop diseases from leaf images using deep learning, with real-time weather-based risk assessment and detailed scan history.

---

## 🚀 Features

- 🔬 **Disease Detection** — Upload a crop leaf image and get instant AI-powered disease diagnosis
- 🌤️ **Weather Risk Assessment** — Real-time weather data to assess crop disease risk in your area
- 📜 **Scan History** — Track all your past scans with disease name, confidence, and severity
- 📊 **Dashboard** — Visual charts showing disease frequency, daily scans, and severity breakdown
- 🔐 **User Authentication** — Secure login and registration system
- 📄 **PDF Report** — Download a detailed report of your scan results

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend/UI | Streamlit |
| AI Model | TensorFlow / Keras (CNN) |
| Database | PostgreSQL (psycopg2) |
| Weather API | OpenWeatherMap |
| PDF Reports | ReportLab |
| Image Processing | Pillow, OpenCV |

---

## 📁 Project Structure

```
CropDiseaseDetection/
├── app.py                  # Main Streamlit application
├── train_model.py          # Model training script
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (do not commit)
├── models/
│   ├── crop_disease_model.h5      # Trained CNN model
│   └── class_indices.json         # Disease class labels
├── utils/
│   ├── db.py               # PostgreSQL database functions
│   ├── auth.py             # Password hashing & validation
│   ├── weather.py          # OpenWeatherMap integration
│   ├── preprocess.py       # Image preprocessing
│   ├── recommendations.py  # Treatment recommendations
│   └── report.py           # PDF report generation
└── dataset/                # Training dataset
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.9+
- PostgreSQL installed and running
- OpenWeatherMap API key (free at [openweathermap.org](https://openweathermap.org/api))

### 1. Clone the Repository
```bash
git clone https://github.com/Dig16Patel/Smart-Crop-Disease-Detection.git
cd Smart-Crop-Disease-Detection
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
pip install psycopg2-binary requests reportlab python-dotenv
```

### 3. Set Up PostgreSQL Database
Run the following in `psql`:
```sql
CREATE DATABASE cropguard_db;
CREATE USER cropguard_user WITH PASSWORD 'cropguard123';
GRANT ALL PRIVILEGES ON DATABASE cropguard_db TO cropguard_user;
```

Then create the required tables:
```sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE scan_history (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    disease_name VARCHAR(100),
    confidence FLOAT,
    severity VARCHAR(20),
    scanned_at TIMESTAMP DEFAULT NOW()
);
```

### 4. Configure Environment Variables
Edit the `.env` file in the project root:
```
DB_HOST=localhost
DB_NAME=cropguard_db
DB_USER=cropguard_user
DB_PASSWORD=cropguard123
DB_PORT=5432
OPENWEATHER_API_KEY=your_api_key_here
```

### 5. Run the App
```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**

---

## 🌱 Supported Crops & Diseases

| Crop | Diseases Detected |
|---|---|
| 🍅 Tomato | Early Blight, Late Blight, Leaf Mold, etc. |
| 🌽 Corn / Maize | Common Rust, Northern Blight, Gray Leaf Spot |
| 🥔 Potato | Early Blight, Late Blight |
| 🍇 Grape | Black Rot, Esca, Leaf Blight |
| 🍎 Apple | Apple Scab, Black Rot, Cedar Rust |

---

## 📸 How to Use

1. **Register / Login** to your account
2. Go to **🔬 Detect Disease** from the sidebar
3. **Upload** a clear photo of the crop leaf
4. View the **diagnosis** — disease name, confidence, severity, and treatment advice
5. Download a **PDF report** of the results
6. Check the **📊 Dashboard** to track your scan history



## 👨‍💻 Authors

- **Dig Patel** — [GitHub](https://github.com/Dig16Patel)

