# List of Backlog Changes
## (and other random thoughts)

### CI/CD
- Setup precommit and github actions

### Better Development
- Poetry
- pyproject.toml file instead of requirements.txt

### Improve Prediction Pipeline
- Batch predictions
    - Get list of videos in file
    - Iterate through and make predictions on each one
    - Save metadata of model as you go
- MLflow tracking
- Dealing with unlabelled data
    - Detection confidence metrics
    - Detection consistency metrics
    - Bounding box confidence metrics

## For Later

### Wrap your head around development tools
- Containers
- Dockers
- Linux ... 🤮

### Incorporate Robotics
- Get your hand on a sensor
- raspberry pi
- ROS2
- Sensor and actuator understanding
- Develop this equivalent in C++?
- Overlaying radar with tracking to map the tracked object in 3d space / on a map

### Modelling
- Fusion modelling with data from multiple sources
- Sensor fusion modelling algorithms
- Pytorch

## Random notes of stuff I wrote down to look into later

- Image preprocessing - what methods are available
- OpenCV (already using), Pillow, Pytorch torchvision
- Google gcp to train models on GPUs and then download it for use on my computer
- OpenMMlab, MMTracking
- AICityChallenge - has image and lidar data for download to experiment with
- DuckVision
- TheOrangeDuck
- Unity

## Next Steps (Priority Order)

### 1. Set Up Tracking Metrics Pipeline
Foundational for everything else.

**Why this matters:**
- I already have unlabeled boat footage to test on
- Metrics are needed to evaluate any model (pretrained or custom)
- Without metrics, can't objectively compare YOLOv8n vs a custom model
- This becomes the validation tool for custom model training later

**Deliverables:**
- Run YOLOv8n on unlabeled boat footage
- Log detection confidence, track consistency, bounding box stability
- Visualize tracking quality without ground truth (confidence heatmaps, track length distributions)
- Integrate with MLflow to compare different tracker configs (BOTSORT params)

### 1.1 Set Up CUDA
Increase training and evaluation speed.

**Why this matters:**
- Currently 1:1 ratio of video to processing length
- Need to speed it up for evaluation and training

### 2. Camera Calibration
**Do this second** - validates the hardware is working correctly.

**Why this matters:**
- The OAK-D is already being used, so verifying calibration makes sense
- Stereo depth accuracy depends on calibration quality
- Quick win that builds confidence in sensor data
- Essential before adding more cameras in Project 2

**What to do:**
- Use DepthAI's factory calibration verification tools
- Test stereo depth accuracy with known-distance objects
- Document any calibration drift or issues
- Save calibration parameters for future reference
- Skip custom calibration routines (factory calibration is usually sufficient)

### 3. Train Custom Model
**Do this third** - now there are metrics to evaluate it.

**Why this matters:**
- The metrics pipeline can validate training progress
- I can compare custom model vs YOLOv8n objectively
- Training on maritime-specific data will improve real-world performance
- Completes Project 1 and addresses "Adapt YOLO for maritime objects"

**Datasets to use:**
- SeaShips (open source maritime dataset)
- SMD (Singapore Maritime Dataset)
- My unlabeled footage for domain adaptation

### 4. Additional Sensors (Later)
Save for Project 2 - requires working baseline first.

---

## Why This Order Makes Sense

**Addresses the constraint:**
Can't test on live water, but can:
- Evaluate tracking on prerecorded footage without ground truth (confidence-based metrics)
- Test calibration indoors with known-distance objects
- Train and validate on prerecorded footage before live deployment

**Aligns with Project 1 goals:**
- ✅ Setup & Calibration (Step 2)
- ✅ Object Detection Pipeline (already done with YOLOv8n)
- ⚠️ Performance Optimization (Step 1 - need metrics first)
- ⚠️ Maritime-specific model (Step 3 - custom training)

**Enables iterative improvement:**
Working metrics allow data-driven decisions about whether custom training is needed.

---

## Practical First Week Plan

**Days 1-2: Extract Metrics Code**
- Create `shared-tracking-metrics/` with existing `VideoMetricsCollector` logic from `boat-tracking-system/`

**Days 3-4: Add Unlabeled Footage Metrics**
Implement metrics that don't require ground truth:
- Detection confidence distribution
- Track length/stability
- Bounding box jitter (frame-to-frame changes)
- Detection consistency (same object detected across frames)

**Days 5-7: Integrate with OAK-D Pipeline**
- Add metrics collection to live camera feed
- Test on prerecorded footage

---

## Decision Point After Step 1

Once metrics are working on unlabeled footage, the data will show:
- Is YOLOv8n good enough? (high confidence, stable tracks)
- Or is custom training needed? (low confidence, unstable detections)

This data-driven decision is better than guessing whether custom training is worth the effort.

**Expected outcome:** YOLOv8n will likely struggle with maritime conditions (glare, waves, distant boats), which will motivate custom model training with clear performance targets to beat.
