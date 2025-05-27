# System Architecture

The proposed system is modular, with separate components for video preprocessing, object detection/segmentation, tracking, pose/action recognition, and event inference.  High-level modules include:

* **Video Ingestion & Preprocessing:** Read input video, extract frames (e.g. via OpenCV), and apply basic preprocessing (resize, color normalization, background subtraction if needed).  Preprocessing may include color jitter and random augmentations to help later domain adaptation (NBA-to-gym).

* **Object Detection & Segmentation:** Detect players and the ball in each frame using a fast object detector (e.g. a person detector or “sports ball” class).  Detected bounding boxes (or points) are used as *prompts* for SAM2 to produce precise segmentation masks per object.  SAM2 (Segment Anything Model 2) can then be run in **video mode** (with memory) to refine masks across frames.  For players, a person detector (YOLOv8/Detectron2) provides initial boxes; for the ball, we can use a small YOLOv8 “sports ball” model or even color-based tracking (e.g. filter orange ball).  SAM2’s video predictor uses these prompts to generate pixel-accurate masks per frame.

* **Multi-Object Tracking:** Track each detected object over time to assign persistent IDs.  We use a *tracking-by-detection* approach: running an efficient tracker (e.g. ByteTrack or BoT-SORT) on the sequence of detected boxes/masks to maintain consistent IDs for players and the ball.  Alternatively, we can leverage SAM-based trackers like **SAM-Track** or **SAMURAI**, which combine SAM segmentation with a transformer tracker (AOT/DeAOT) for mask propagation.  In either case, the result is a set of object tracks (player IDs) across the video frames.

* **Pose Estimation & Action Classification:** For each tracked player, apply a pose-estimation model (e.g. Detectron2 Keypoint R-CNN or MMPose) to extract skeleton joints in every frame.  These 2D poses (or the raw cropped player video) feed into a temporal action-recognition model (e.g. a 3D CNN or Transformer in MMAction2) to classify actions like *shoot, pass, dribble, move off-ball, rebound*, etc.  We will likely fine-tune a pre-trained video classification network (such as an R(2+1)D or SlowFast model initialized on Kinetics-400) on labeled basketball action clips.  Pose features can also aid rule-based detection (e.g. raised arms for shooting).

* **Event Detection & Stats:** Combine the tracked data to infer game events.  For example, a *shot attempt* is detected when a player’s shooting action is recognized and the ball’s trajectory goes toward the rim.  A *made shot* can be flagged when the ball’s position crosses through the hoop (we may detect the hoop via a YOLO model and check if the ball passes between the rim bounds).  *Passes* occur when the ball leaves one player’s control and is caught by another (track ball-to-player handover).  *Rebounds* are inferred when the ball bounces off the rim/backboard and possession changes.  *Off-ball movement* is simply any significant motion by a player not controlling the ball.  All events are logged with timestamps and involved player IDs.

* **Output Generation:** The system outputs an annotated video (overlaying bounding boxes, masks, skeletons and labels on frames) for visual verification.  It also exports a JSON file containing structured stats and events: shot attempts/makes per player, pass sequences (from player A to B), rebound events, etc.  This JSON can feed into dashboards or analytics tools.

## SAM2-Based Segmentation and Tracking

&#x20;*Illustrative output: SAM2 can generate detailed segmentation masks for objects (e.g. ball and players) in a sports scene.* SAM2 provides promptable, high-quality segmentation masks on video frames.  In our pipeline, we will use SAM2’s **video predictor**: given one or more object prompts (bounding boxes or points in a key frame), SAM2’s memory-based architecture propagates and refines object masks through the sequence. For example, we first detect the basketball in frame 0 (via YOLO or color filtering) and supply that box to SAM2 as a prompt.  SAM2 then outputs pixel masks of the ball in each subsequent frame, along with per-frame bounding boxes.  This gives us precise ball segmentation and coordinates automatically. We do the same for each player (e.g. use YOLO to detect all players in key frames, then feed to SAM2).

Because SAM2 tracks objects via its memory attention, it can handle occlusions and movement better than frame-by-frame segmentation.  However, SAM2 alone does not assign consistent IDs, so we combine it with a tracker.  A proven approach is “SAM-Track”: use SAM for segmentation and a transformer-based tracker (AOT/DeAOT) for IDs.  In practice, we can also feed SAM2’s output boxes into a standard tracker like ByteTrack (via Ultralytics YOLO’s tracking APIs).  Either way, the result is a set of object IDs (players 1…N and the ball) with segment masks over time.  These masks can also improve tracking by providing more accurate IoU for association.

By combining detection → SAM2 segmentation → tracking, the pipeline ensures each player and the ball are isolated in every frame.  For instance, SAM2’s JSON output gives per-frame `(object_id, bbox, mask)` for the ball, which we integrate into our trackers.

## Player & Ball Detection and Tracking

* **Player Detection:** Use a person/object detector (e.g. YOLOv8 or Detectron2) to find players in each frame.  This yields initial bounding boxes and confidences.  Optionally, run a keypoint detector on top of these (Detectron2’s Keypoint R-CNN) to get body joints, which will feed the action analysis stage.  The detected boxes are sent to the tracker or used as SAM prompts.
* **Ball Detection:** The basketball is small and fast.  We propose a hybrid approach:

  * **YOLO / ML detection:** Train or use a pre-trained “sports ball” detector (e.g. YOLOv8’s open-vocabulary models) to get ball boxes.  Because YOLO may miss the ball or yield false positives (especially in clutter), we add a **validation** step (e.g. a small classifier or heuristic checking color/shape) to confirm detections.  - **Classical tracking:** Alternatively, simple color filtering (track the orange ball by HSV threshold) can provide a strong cue in many indoor courts if lighting is controlled.  - **Specialized model:** We can also consider a dedicated ball-tracking CNN (e.g. TrackNetV3) which uses multi-frame cues and background subtraction to locate the ball.  If used, that model’s output box is fed to SAM2 for segmentation.
* **Tracking Algorithm:** After detection, apply a robust multi-object tracker.  Ultralytics YOLO’s tracker supports ByteTrack, BoT-SORT, etc.. We will use ByteTrack (high recall) for both players and ball.  The tracker links detections across frames by matching boxes (and optionally appearance features), giving each player a unique ID.  For added robustness, SAM2 segmentation masks can refine the tracking IoU.
* **SAM2 Integration:** At each key frame (or continuously), we provide the detector’s boxes as prompts to SAM2’s video model.  This yields smooth masks even when detectors flicker.  In practice, we may re-prompt every few seconds to correct drift. SAM2’s output JSON (per-object track of bboxes and mask areas) is merged with the tracker IDs.

By the end of this stage, every player track (ID) has a time series of positions, masks, and optionally poses, and the ball has its own track.  This solves **Requirement 1** (player tracking) and **Requirement 3** (ball tracking).

## Pose Estimation & Action Recognition

Once players are localized and tracked, we analyze their actions:

* **Pose Estimation:** Run a 2D pose estimator (e.g. MMPose HRNet, OpenPose, or Detectron2 Keypoint R-CNN) on each player crop to get joint coordinates (e.g. shoulders, elbows, knees).  These 2D skeletons provide features for action classification (e.g. angle of arm for shooting).  No training is needed for standard poses (COCO-trained models are sufficient).
* **Action Classification:** For each player track, we split the video into temporal clips (e.g. 1–2 second segments around interesting intervals) and classify the action.  We consider key basketball actions: *Dribble, Pass, Shoot, Rebound, Move off-ball*.  Two strategies can be used in parallel:

  1. **Video CNN (MMAction2):** Train a spatio-temporal network on labeled basketball clips.  For example, an R(2+1)D model pre-trained on Kinetics-400 can be fine-tuned on a basketball dataset.  The SpaceJam action dataset (∼32k clips of single-player actions) labels dribble, pass, shoot, etc..  We would similarly annotate a dataset (possibly using synthetic augmentation or manual labeling) and train the 3D CNN to recognize each action.
  2. **Pose-based Heuristics:** Use pose trajectories and ball interaction to detect actions.  E.g., if a player’s hands raise above head with the ball, it likely is a *shot*; if hands are at waist with ball bouncing, *dribble*; if ball leaves hand toward another player, *pass*.  Combining pose with ball relative motion (from tracking) can rule-based tag events.
* **Integration:** The classifier outputs per-frame (or per-clip) action labels for each player.  These are mapped to our required events.  For instance, a “shoot” classification plus ball moving toward hoop → shot attempt. A “pass” plus ball trajectory to another player → pass event.

We will employ PyTorch-based toolkits: **MMAction2** (for video model training) and **Detectron2/MMPose** (for pose) as recommended platforms.  We may also consider transformer-based video models (e.g. ViViT, VideoMAE) if needed.

## Preprocessing, Augmentation & Domain Adaptation

Because we train on NBA footage but deploy on amateur gym footage, we must bridge the domain gap:

* **Augmentation:** Apply heavy augmentations during training: varying brightness/contrast, random crops/zoom, noise, blur and low-resolution simulation to mimic poor camera quality.  Also color-jitter the court colors and player uniforms to generalize beyond polished TV broadcasts.  See \[24] for handling video quality differences (lower resolution, motion blur).
* **Synthetic Data & Transfer:** If possible, generate synthetic training examples: e.g. use sports video rendering or cycleGAN-like style transfers to make NBA frames look “gym-like.”  Alternatively, fine-tune models on whatever in-domain video we can collect.
* **Court Alignment:** Basketball courts have known markings; we can use homography (detect the free-throw or 3-point line) to normalize camera angle and scale.  This helps tracking and reduces viewpoint variation.
* **Multi-crop Testing:** For stability on low-quality feeds, consider applying the detector/predictor on overlapping patches or lower FPS (skip frames), then fuse results.

Domain adaptation can also involve unsupervised methods, but a simpler approach is to manually label a small set of gym videos (using tools like Roboflow or Supervision) and include them in training or fine-tuning.

## Tools, Frameworks & Libraries

We recommend the following stack:

* **PyTorch** – primary DL framework.
* **Detectron2** – for baseline person detection, instance segmentation (mask R-CNN), and keypoint (pose) models.
* **YOLOv8 (Ultralytics)** – for fast object detection/tracking (person and ball classes); built-in trackers (ByteTrack, BoT-SORT).
* **SAM2 Implementation** – use Meta’s `segment-anything-2` repository.  For ease of use, leverage high-level interfaces (e.g. \[Roboflow’s SAM2 tutorial]\[4] or \[Sieve’s API]\[32]) that wrap the `build_sam2_video_predictor` model.
* **OpenCV** – for video I/O, drawing annotations, and utility processing (color filtering, homography, optical flow if needed).
* **MMAction2** – for building and training the action recognition model (it has video data loaders and many architectures).
* **MMPose** – for 2D human pose estimation.
* **MMTracking or Custom Tracker** – we can use Ultralytics’ tracking via YOLO (easy API), or MMTracking for more customization.
* **Supervision** – a Python package (by Roboflow) useful for visualizing segmentation/tracking outputs.
* **Dashboard / JSON Handling:** Use Pandas/NumPy for stats aggregation and libraries like Plotly Dash or Grafana (out of scope now) for eventual dashboards.

These tools are all PyTorch-compatible and work on local GPUs.  Video processing can be done batch-wise (frame by frame or small clips) to fit GPU memory.

## Phased Implementation Plan

| **Phase**                              | **Tasks**                                                                                                                                                                                                                                                                     | **Deliverables**                                                                          | **Notes**                                                                                             |
| -------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| **1. Data Collection & Annotation**    | Gather NBA video clips. Use SAM2 (with supervision) or Roboflow to annotate a small dataset: label player masks, ball masks, hoop.  Extract action clips (shoot, pass, etc.) and label them.                                                                                  | Labeled training set: player/ball bounding boxes and masks, plus action-classified clips. | Start with \~100 games. Augment data (flip, crop).  Use ball prompt examples to seed SAM2.            |
| **2. Detection & Tracking Prototype**  | Train a person detector (e.g. YOLOv8) on labeled frames. Train a ball detector (or use color filter). Integrate with ByteTrack to track objects.  Add SAM2: feed detector boxes to SAM2 and output masks. Verify that players and ball get consistent IDs.                    | Pipeline that takes video → player tracks + ball track + annotated video overlay.         | Evaluate on held-out videos; adjust detection thresholds.                                             |
| **3. Action Recognition Module**       | Train or fine-tune a 3D CNN (R(2+1)D, SlowFast, or Transformer) on the labeled basketball action clips.  Validate that it can classify dribble, pass, shoot, etc., for a single player clip.  Integrate this model: for each player track, run sliding-window classification. | Action recognition model + code to label each tracked player sequence with actions.       | Use MMAction2 for experimentation. Optionally incorporate pose features or optical flow for accuracy. |
| **4. Event Inference & JSON Output**   | Develop logic to combine tracks and actions into events: detect shot attempts/makes (using action label + ball-hoop geometry), passes, rebounds, off-ball movement.  Implement JSON schema and fill it with events (player IDs, timestamps).  Generate final annotated video. | JSON files per video with stats and events; example annotated videos.                     | Test accuracy of event detection. Refine rules (e.g. ball bounce detection for rebounds).             |
| **5. Domain Adaptation & Fine-tuning** | Collect a small dataset of gym footage (even just webcam games).  Fine-tune detectors (player/ball) on this data.  Augment training with synthetic gym-like distortions.  Possibly distill SAM2 prompts on gym images.                                                        | Adapted models (player/ball detector, action model) that work on gym video.               | Evaluate on gym videos and iterate. Compare with NBA baseline.                                        |
| **6. Optimization & Deployment**       | Profile pipeline on target GPU (e.g. RTX 3060).  Optimize by using smaller models or lower frame rates as needed.  Convert models to half-precision or TorchScript if supported.  Implement batch/frame skipping optimizations.  Prepare integration to dashboard backend.    | Final code running in near real-time on local GPU; documentation on usage.                | Consider pruning or quantization if very slow.  Monitor memory/CPU load.                              |

Each phase builds on the previous.  We can deploy intermediate versions for testing and user feedback.

## GPU Optimization Tips

Running this pipeline on a single GPU requires efficiency:

* **Model sizing:** Use the smallest adequate models.  For SAM2, the *tiny* or *small* variant can yield \~40 FPS on high-end GPUs.  On a 3060 or 40-series, expect lower rates; consider using SAM2 *hiera\_tiny* or *small* instead of *large*. For detection and action models, use lightweight versions (YOLOn instead of YOLOx or pruned 3D CNNs).
* **Precision and Batching:** Run inference in FP16 (half-precision) when possible to double throughput.  Use `torch.cuda.amp` or compile models with TorchScript/ONNX for faster inference.  Process frames in small batches where possible.
* **Frame Skipping:** For tracking, you may not need to process every frame at full resolution.  E.g. run detection every 2–3 frames and interpolate tracks in between, or use tracker’s motion model. This cuts compute by \~2x.
* **Resolution:** Crop tightly around court region and resize to the model’s input (e.g. 640×360) instead of full HD.  Smaller images speed up all models.
* **Asynchronous Pipeline:** Use separate threads/processes for reading frames, running models, and drawing outputs.  For example, one GPU can handle segmentation while another (or CPU thread) draws annotations or prepares the next batch.
* **Profiling:** Monitor GPU utilization (e.g. NVIDIA-smi).  If GPU is idle while waiting for data, increase batch size or prefetch more frames. If GPU memory is the limit, reduce batch size or model size.

Benchmark each component in isolation (e.g. how many frames/s does SAM2 tiny do on your GPU) and tune accordingly.

## Citations

This plan leverages recent advances in video segmentation and tracking (SAM2), multi-object tracking, and sports action recognition.  We cite relevant techniques for segmentation/tracking pipelines, basketball action datasets, and practical tracking demos. These references support our architecture choices and strategies.
