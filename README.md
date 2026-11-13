# NetGuard-AI
### Real-Time Network Intrusion Detection System Using Hybrid Machine Learning and Explainable AI

NetGuard-AI is a machine learning-based Network Intrusion Detection System (NIDS) designed to identify malicious network traffic, detect anomalous activity, and provide interpretable explanations for prediction decisions.

The system combines **XGBoost supervised classification, Isolation Forest anomaly detection, and SHAP explainability** in a Flask-SocketIO web application. It provides a monitoring dashboard with attack alerts, geolocation visualization, model evaluation results, and configurable detection thresholds.

---

## 1. Project Overview

Traditional signature-based Intrusion Detection Systems (IDS) primarily depend on predefined attack patterns and manually updated rules. These systems may struggle to identify previously unseen attacks and evolving network threats.

NetGuard-AI addresses these limitations through a hybrid machine learning approach that combines supervised and unsupervised learning.

The system is developed and evaluated using the CIC-IDS-2017 dataset.

### Key Highlights

- Trained and evaluated using CIC-IDS-2017 network traffic data.
- Evaluated 13 machine learning and deep learning models under identical conditions.
- Selected XGBoost using a multi-criteria model evaluation framework.
- Achieved a composite model selection score of 0.9948.
- Integrated Isolation Forest for anomaly-based detection.
- Uses SHAP TreeExplainer for prediction-level explainability.
- Provides a Flask-SocketIO-based monitoring dashboard.

---

## 2. Key Features

### Hybrid Intrusion Detection

Combines two complementary detection approaches:

- **XGBoost:** Supervised classification of network traffic into benign and attack categories.
- **Isolation Forest:** Unsupervised anomaly detection to identify unusual traffic patterns that may indicate previously unseen threats.

The outputs are combined through a weighted scoring mechanism with configurable detection thresholds.

### Explainable AI Using SHAP

NetGuard-AI integrates SHAP (SHapley Additive exPlanations) to improve the interpretability of machine learning predictions.

- Uses SHAP TreeExplainer.
- Identifies the top three contributing features for individual predictions.
- Helps analysts understand why a particular network flow was classified as malicious or benign.
- Provides feature importance visualizations.

### Multi-Criteria Model Evaluation

Instead of selecting a model based only on accuracy, the project uses a weighted composite evaluation framework.

| Evaluation Metric | Weight |
|---|---:|
| Recall | 30% |
| F1-Score | 25% |
| False Alarm Rate (FAR) | 20% |
| AUC-ROC | 15% |
| Inference Speed | 10% |

XGBoost was selected as the best-performing model under this evaluation framework, achieving a composite score of 0.9948.

### Web Dashboard

The web interface provides:

- Attack monitoring and alert feed.
- Network traffic prediction interface.
- Geographic visualization of detected traffic.
- Model evaluation and performance visualizations.
- Adjustable detection thresholds.
- Historical prediction records using SQLite.
- User registration and login with bcrypt password hashing.

---

## 3. System Architecture

The system follows a modular architecture consisting of five major components.

1. **Traffic Generator:** Reads network traffic flows from CSV files and emits Socket.IO events.
2. **Flask-SocketIO Server:** Handles incoming events, application requests, and authentication.
3. **Inference Engine:** Performs hybrid detection using XGBoost, Isolation Forest, and SHAP.
4. **SQLite Database:** Stores prediction records for historical analysis.
5. **Browser Dashboard:** Displays alerts, network activity, geographic information, and performance visualizations.

### System Workflow

```text
CIC-IDS-2017 Traffic Dataset
            |
            v
    Traffic Generator
            |
            v
    Flask-SocketIO Server
            |
            v
      Inference Engine
            |
       +----+----+
       |         |
       v         v
    XGBoost   Isolation
              Forest
       |         |
       +----+----+
            |
            v
     Hybrid Detection
            |
       +----+------+
       |           |
       v           v
   Prediction     SHAP
                Explanation
       |           |
       +-----+-----+
             |
             v
       SQLite Database
             |
             v
       Web Dashboard
             |
        +----+-----+
        |          |
        v          v
    Live Alerts  GeoIP Map
```

---

## 4. Dataset

**Dataset:** CIC-IDS-2017

The project presentation reports that the system was trained and evaluated using approximately 2.83 million network traffic flows covering 15 attack types.

The dataset provides network traffic features for distinguishing benign and malicious network activities.

### Data Preprocessing

A four-stage preprocessing pipeline was implemented with the following objectives:

- Cleaning and preparing network traffic data.
- Handling class imbalance.
- Scaling numerical features.
- Selecting discriminative features.

The feature selection process reduced the original feature set from 78 features to 30 selected features.

**Note:** The dataset is not included in this GitHub repository because of its large file size. The local project uses a dataset file named `CICIDS2017_subset.csv`.

To run the project, obtain the appropriate CIC-IDS-2017 data and place the required CSV file in the `data/` directory.

---

## 5. Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python 3.11 |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn, XGBoost |
| Deep Learning | TensorFlow, Keras |
| Explainable AI | SHAP (TreeExplainer) |
| Backend | Flask 3.x, Flask-SocketIO |
| Database | SQLite, SQLAlchemy |
| Authentication | bcrypt |
| Model Persistence | Joblib |
| Frontend | HTML, CSS, JavaScript |
| Visualization | Chart.js, Leaflet.js |
| Templates | Jinja2 |
| Development Environment | Visual Studio Code |

---

## 6. Model Performance

The project presentation reports the following performance results for the selected detection system.

| Evaluation Metric | Reported Result |
|---|---:|
| Detection Accuracy | 99.905% |
| Recall (Sensitivity) | 0.9998 |
| F1-Score | 0.9972 |
| False Alarm Rate | 0.11% |
| AUC-ROC | 1.0000 |
| Inference Time per Flow | 0.00065 ms |

*Note: These are the results reported in the project presentation. Actual performance may vary depending on the dataset split, hardware, traffic distribution, and execution environment. Inference latency should be independently benchmarked under documented hardware and measurement conditions.*

---

## 7. Project Structure

```text
NetGuard-AI/
│
├── app.py
├── train.py
├── inference.py
├── generator.py
├── generate_shap.py
├── utils.py
├── commands.md
├── requirements.txt
├── setup.bat
├── .gitignore
│
├── models/
│   ├── scaler.joblib
│   ├── supervised_best.joblib
│   ├── unsupervised_iso.joblib
│   ├── shap_explainer.joblib
│   ├── feature_keys.json
│   ├── metrics.json
│   ├── comparison_results.json
│   ├── comparison_plot.png
│   └── feature_importance_plot.png
│
├── static/
│   ├── css/
│   │   └── style.css
│   └── images/
│       └── flowchart.png
│
└── templates/
    ├── base.html
    ├── index.html
    ├── login.html
    ├── register.html
    ├── analysis.html
    ├── dataset.html
    ├── model_evaluation.html
    └── prediction.html
```

**Note:** The `data/`, `venv/`, `GeoLite2-City.mmdb`, and local `nids.db` files are excluded from the repository.

---

## 8. Installation and Execution

### Prerequisites

- Python 3.11
- Git
- pip
- Windows, Linux, or another compatible Python environment

### Step 1: Clone the Repository

```bash
git clone https://github.com/NavjotKaurCheema/NetGuard-AI.git
cd NetGuard-AI
```

### Step 2: Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

On Linux:

```bash
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure Dataset

Create a `data` directory and place the required CIC-IDS-2017 CSV file inside it.

Expected dataset filename:

```text
data/CICIDS2017_subset.csv
```

Ensure that the dataset path and expected feature columns match the project configuration.

### Step 5: Configure GeoIP Database

The GeoIP database is not included in this repository.

Obtain the appropriate MaxMind GeoLite2 City database according to its licensing and access requirements, and place the database file in the project root if required by the application.

Expected filename:

```text
GeoLite2-City.mmdb
```

### Step 6: Run the Application

Start the Flask application:

```bash
python app.py
```

Open the local application URL displayed in the terminal.

*Note: Refer to `setup.bat` and `commands.md` for additional project-specific configuration and setup instructions.*

---

## 9. Application Usage

The typical application workflow is:

1. **Register / Login:** Create an account or log in through the authentication interface.
2. **Start Traffic Feed:** Start the CSV-based traffic generator.
3. **Live Detection:** Network flows are processed through the hybrid detection engine.
4. **View Alerts:** Monitor detected attacks through the alert feed and geographic map.
5. **Adjust Thresholds:** Tune detection parameters using dashboard controls.
6. **Analyse History:** Review stored predictions and model evaluation visualizations.

---

## 10. Challenges Addressed

Some of the major development challenges included:

- **Dataset scale and class imbalance:** Handling approximately 2.83 million network flows and multiple attack categories.
- **Deep learning training time:** Training computationally intensive models such as LSTM, GRU, and CNN-BiLSTM.
- **SHAP computation overhead:** Using subsampling for global explainability analysis and prediction-level SHAP during inference.
- **Real-time GeoIP visualization:** Handling synthetic CSV IP addresses and preventing duplicate map markers.

---

## 11. Future Scope

The project can be extended in several directions:

- **Live Packet Capture:** Replace CSV-based traffic replay with live network traffic capture using tools such as CICFlowMeter.
- **Multi-Class Classification:** Extend binary benign/attack classification to identify individual attack categories such as DoS, DDoS, Botnet, and Web Attacks.
- **Federated Learning:** Train intrusion detection models across multiple organizations without sharing raw network traffic.
- **Online Learning:** Incorporate incremental learning to adapt to evolving attack patterns and concept drift.

---

**NetGuard-AI — Making Network Intrusion Detection Intelligent, Explainable, and Real-Time.**
