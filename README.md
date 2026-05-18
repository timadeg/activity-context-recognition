# Activity Context Recognition

Machine learning project for classifying human physical activities from smartphone sensor data. The model learns to distinguish between activities such as sitting, walking, running, driving, and stair climbing using motion, orientation, light, and sound readings captured from a mobile device.

## Dataset

`activity_context_tracking_data.csv` contains time-series sensor readings labelled with the activity being performed. Each row is a single observation.

**Features (18 columns):**

| Column | Description |
|---|---|
| `_id` | Unique observation ID |
| `orX`, `orY`, `orZ` | Device orientation (x, y, z axes) |
| `rX`, `rY`, `rZ` | Rotation vector |
| `accX`, `accY`, `accZ` | Accelerometer readings |
| `gX`, `gY`, `gZ` | Gravity sensor readings |
| `mX`, `mY`, `mZ` | Magnetometer readings |
| `lux` | Ambient light level |
| `soundLevel` | Ambient sound level |
| `activity` | **Target variable** — labelled activity |

**Activity classes:** AscendingStairs, ClimbingDownStairs, ClimbingUpStairs, DescendingStairs, Driving, Jogging, Lying, MountainAscending, MountainDescending, Running, Sitting, Standing, Walking.

## Files

- `activity_context_tracking_data.csv` — the dataset
- `eda.py` — exploratory data analysis script
- `machine learning.ipynb` — model training and evaluation notebook

## Getting started

Clone the repo and install the usual data science stack:

```bash
git clone https://github.com/timadeg/activity-context-recognition.git
cd activity-context-recognition
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

Run the EDA script:

```bash
python eda.py
```

Or open the notebook:

```bash
jupyter notebook "machine learning.ipynb"
```
