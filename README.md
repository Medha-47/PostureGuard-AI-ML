# PostureGuard-AI 

A real-time AI tool that uses computer vision to detect slouching and notify the user to sit straight. It uses YOLOv8-Pose to track body keypoints and calculates a normalized distance to ensure accuracy regardless of camera distance.



## How it Works
The system benchmarks your "perfect posture" during a 5-second calibration phase. It then monitors the ratio between your ear, shoulder, and hip. 

**The Formula:**
The system uses a scale-invariant ratio to detect slouching:
$$Normalized Distance = \frac{Distance(Ear, Shoulder)}{Distance(Shoulder, Hip)}$$

## Features
* **Adaptive Calibration:** Calibrates to your specific height and seating position.
* **Live Visuals:** Color-coded status (Green = Good, Red = Severe Slouch).
* **Desktop Notifications:** Sends an alert if you slouch for more than 30 seconds.
* **Progress Tracking:** Saves every session to `posture_log.csv`.

   

##  Setup and Installation

### 1. Clone this repository
```bash
git clone [https://github.com/medha-47/PostureGuard-AI.git]
   (https://github.com/medha-47/PostureGuard-AI.git)
```

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the application
```bash
python app.py
```


