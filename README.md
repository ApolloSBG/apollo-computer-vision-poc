# Apollo Computer Vision POC

An experimental computer vision project exploring how AI can be applied to sports sponsorship visibility and exposure analysis in broadcast footage.

The project is being developed as part of Apollo Sports Business Group's experimentation with data science and AI-driven sponsorship intelligence. It applies techniques including frame extraction, object detection, transfer learning, image annotation, model fine-tuning, tracking, and evaluation on unseen broadcast footage.

## Project Objective

The goal is to investigate whether computer vision models can automatically identify and eventually track sponsor brands and advertising assets appearing during sports broadcasts.

The current proof of concept focuses on football match footage and explores a pipeline for:

- Extracting frames from broadcast video
- Detecting objects using pretrained YOLO models
- Creating custom sponsor-brand annotations with CVAT
- Fine-tuning YOLO for brand-specific object detection
- Testing the trained model on unseen match frames
- Exploring object tracking across video sequences
- Building toward automated sponsorship exposure measurement

## Current POC

The first custom training experiment uses manually annotated broadcast frames containing five sponsor brands:

- DORADOBET
- Toña
- STIHL
- ISUZU
- Chevrolet

A YOLO11n model is fine-tuned on these annotations and subsequently tested against frames that were not included in the training set.

This initial dataset is intentionally small and is designed to validate the end-to-end technical workflow rather than produce a production-ready detection model.

## Development Roadmap

The project will progressively explore:

**Pretrained Detection → Custom Annotation → Transfer Learning → Data Augmentation → Brand Detection → Tracking → Exposure Intelligence**

Future iterations will increase dataset size and diversity, introduce separate training/validation/test datasets, apply realistic image augmentation, improve model generalization, and evaluate performance across different camera angles, distances, lighting conditions, and matches.

The longer-term research objective is to explore how detected sponsorship assets can be transformed into structured exposure data for sponsorship auditing and intelligence applications.

## Technology

Python · Ultralytics YOLO · PyTorch · OpenCV · CVAT

## Repository Structure

`src/` contains the Python scripts used for frame extraction, detection, custom-model testing, and tracking.

Raw match footage, extracted frames, annotation datasets, trained model weights, and generated training outputs are intentionally excluded from this public repository.

## Status

🚧 **Proof of Concept — Active Development**

The current model and results are experimental. Performance from the initial five-frame training dataset should not be interpreted as representative of production-level detection accuracy.
