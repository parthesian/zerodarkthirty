Here’s a concrete, **step-by-step plan** for obtaining NBA footage and preparing it for use in your pipeline (detection, tracking, and action classification). The process includes sourcing footage, cutting clips, extracting frames, and generating annotations efficiently with tools like **Roboflow**, **Supervision**, and **SAM2**.

---

## 🏀 Phase 1: Acquire & Prepare NBA Footage

### **1.1. Footage Collection (Legal + Practical)**

**Sources (choose based on priority):**

* **YouTube (easiest):**

  * Use full-game replays or highlight compilations from official NBA channels or third parties.
  * Examples: “NBA Full Game Highlights”, “NBA 4K Highlights”, “FreeDawkins”.

* **NBA API / Stats Website (Official Source):**

  * The NBA offers access to some video content for research under its API. You can fetch play-by-play data and match it with video.

* **Dataset Options (if available):**

  * [SportsMOT](https://github.com/SportsMOT) – person/ball tracking in basketball.
  * [SpaceJam Dataset](https://github.com/amir-abdi/spacejam) – 30k labeled clips of player actions (dribble, shoot, pass, etc.).

**Tools:**

* Use `yt-dlp` to download YouTube videos:

```bash
yt-dlp "<youtube_url>" -o "nba_raw/%(title)s.%(ext)s"
```

**Deliverable:** 10–30 full game videos or highlight compilations at 720p+ quality.

---

## 🧩 Phase 2: Clip Cutting & Frame Extraction

### **2.1. Isolate Action Clips**

You don’t need full games processed start-to-finish. Instead, isolate sequences that likely contain:

* Shot attempts
* Passes
* Dribbling
* Off-ball movement
* Rebounds

**Automated Filtering (optional):**

* Use `OpenCV` to split videos into 5–10 second overlapping clips.
* Add motion thresholding or detect scoreboard changes to auto-filter relevant plays.

**Script Example:**

```python
import cv2

def extract_clips(video_path, out_dir, clip_len=5, stride=2):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frames_per_clip = fps * clip_len
    stride_frames = fps * stride

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    i = 0
    while i + frames_per_clip < total:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(f"{out_dir}/clip_{i}.mp4", fourcc, fps, (int(cap.get(3)), int(cap.get(4))))
        for _ in range(frames_per_clip):
            ret, frame = cap.read()
            if ret:
                out.write(frame)
        out.release()
        i += stride_frames
```

### **2.2. Extract Key Frames for Annotation**

For each clip (or every N frames), export frames as `.jpg`:

```bash
ffmpeg -i clip_1.mp4 -vf fps=1 frames/clip_1/frame_%03d.jpg
```

---

## 🧠 Phase 3: Annotation & Labeling Strategy

### **3.1. Tools for Labeling**

Use tools that support **segmentation masks** and **tracking IDs**.

#### ✅ Best Option: **Roboflow Annotate**

* Import frames or clips
* Annotate with bounding boxes or segmentation masks
* Label player, ball, hoop
* Export in COCO, YOLO, or custom formats

**Alternative:**

* [CVAT](https://github.com/opencv/cvat) – advanced, supports object tracking
* [Supervisely](https://supervise.ly) – cloud tool for segmentation
* [Label Studio](https://labelstud.io) – open-source labeling tool

### **3.2. Boost Efficiency with SAM2 (Semi-Automated Annotation)**

Use **SAM2** (Segment Anything Model 2) to segment players and the ball using just bounding box prompts.

**Workflow:**

1. Detect players/ball with YOLOv8 (coarse boxes)
2. Prompt SAM2 with each box on a key frame
3. Propagate masks using SAM2’s video memory mode
4. Use `Supervision` or `Roboflow` to correct masks

**Code Sample with SAM2 + Supervision:**

```python
from supervision import VideoSink, annotate_frame
from segment_anything import SamPredictor, sam_model_registry
# load SAM2 model + run on a box prompt
# generate masks per object -> overlay + save
```

### **3.3. Label Action Classes**

For each clip or player track:

* Tag with action: `dribble`, `pass`, `shoot`, `rebound`, `off_ball`
* Use a spreadsheet or JSON index like:

```json
{
  "clip_id": "clip_003",
  "players": [
    { "id": 2, "action": "dribble" },
    { "id": 5, "action": "off_ball" }
  ]
}
```

You’ll later use this to train action classifiers or evaluate rule-based events.

---

## 🗃 Phase 4: Organize & Export

### **Directory Structure**

```
/nba_data/
  /raw_videos/
  /clips/
    clip_001.mp4
  /frames/
    clip_001/
      frame_001.jpg
  /annotations/
    clip_001.json
  /labeled_data/
    images + masks (via Roboflow export)
```

### **Export Format**

Use **COCO**, **YOLOv8-seg**, or a custom JSON schema:

* `object_id`
* `bbox` and `segmentation`
* `frame_index`
* `class`: player, ball, hoop
* `action` (if known)

---

## Summary Timeline

| **Week** | **Goals**                                       |
| -------- | ----------------------------------------------- |
| 1        | Download NBA videos (YouTube or datasets)       |
| 2        | Auto-slice clips and extract frames             |
| 3        | Annotate clips using SAM2 + Roboflow            |
| 4        | Label actions per player per clip               |
| 5+       | Start training detection/tracking/action models |

---

Would you like a working Jupyter notebook or script that:

* Runs YOLOv8 to auto-detect players and balls
* Prompts SAM2 and propagates segmentation
* Saves annotated clips for Roboflow upload?

Let me know and I’ll generate it.
