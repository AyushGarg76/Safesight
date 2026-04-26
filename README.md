# 🪖 Safesight 
A project under the course UCS532: Computer vision (3W13)

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Faster%20R--CNN-orange?style=flat-square&logo=pytorch)
![Flask](https://img.shields.io/badge/Flask-API%20Backend-lightgrey?style=flat-square&logo=flask)
![React](https://img.shields.io/badge/React-Vite%20Frontend-61DAFB?style=flat-square&logo=react)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

*An end-to-end computer vision system for detecting helmet compliance in video — without YOLO.*
</div>

## Table of Contents

- [Overview](#overview)
- [Motivation](#motivation)
- [Problem Statement](#problem-statement)
- [Key Idea](#key-idea)
- [System Architecture](#system-architecture)
- [Pipeline Walkthrough](#pipeline-walkthrough)
- [Core Logic: Helmet Reasoning Engine](#core-logic-helmet-reasoning-engine)
- [Mathematics](#mathematics)
- [Training Architecture](#training-architecture)
- [API Reference](#api-reference)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Evaluation](#evaluation)
- [Strengths & Limitations](#strengths--limitations)
- [Future Work](#future-work)

## Overview

SafeSight is a *multi-stage computer vision pipeline* that processes uploaded videos to detect helmet-compliance violations. It is built without YOLO — instead using *Faster R-CNN* (ResNet-50 FPN backbone) combined with a custom rule-based spatial reasoning engine that infers whether a person is wearing a helmet.

The system is not just a model. It includes:

- A trained *Faster R-CNN* detector for helmet, head, and person
- A *Helmet Reasoning Engine* using IoU-based spatial logic
- A *Flask REST API* with background job processing
- A *React / Vite* frontend for upload, progress tracking, and results
- Optional *image enhancement* pre-processing for low-quality footage

## Motivation
Industrial environments like warehouses and factories are high-risk zones where strict safety compliance—such as wearing helmets—is essential. However, manual monitoring is often inconsistent, error-prone, and not scalable across large facilities. As a result, safety violations frequently go unnoticed until accidents occur.

SafeSight automates safety monitoring using computer vision to detect helmet compliance in real time. It enables continuous surveillance through existing camera systems, reduces reliance on manual supervision, and improves overall workplace safety efficiently.



## Problem Statement

In safety-critical environments — construction sites, factories, roads — monitoring helmet compliance from CCTV or recorded video manually is:

- *Slow* — hours of footage per inspector
- *Error-prone* — fatigue causes missed violations
- *Not scalable* — grows linearly with camera count

The detection problem itself is hard:

- The same person appears across hundreds of frames
- Helmets can be partially occluded
- Lighting conditions vary drastically
- Heads are often small relative to the full frame
- Raw detections must be converted into meaningful, timestamped violation events

*Goal:* Build an automated system that detects violations, annotates the video, and returns a structured report through an API — fast enough for practical use and explainable enough to debug.





## Key Idea

> Instead of detecting helmets with YOLO, use *Faster R-CNN + Spatial Reasoning*.

The model is trained to detect three objects:

| Class | Meaning |
|---|---|
| helmet | A helmet visible in the frame |
| head | A bare head (no helmet visible) |
| person | A person's full body |

The *Helmet Reasoning Engine* then interprets these detections using two complementary rules, combining them to produce a final violation decision per person per frame.


## System Architecture

### High-Level System Design

mermaid
flowchart LR
    U[👤 User] --> FE[React / Vite\nFrontend]
    FE -->|POST /api/upload| API[Flask API\nServer]

    API --> JM[Job Manager\nuuid + threading]
    JM --> VP[Video Processor\nBackground Thread]

    VP --> FS[Frame Sampler\nframe_idx % FRAME_SKIP]
    FS --> PRE[Pre-processing\nBGR→RGB, Resize, Normalize]
    PRE --> DET[Faster R-CNN\nResNet-50 FPN]

    DET --> LOGIC[Helmet Reasoning\nEngine]
    LOGIC --> AGG[Violation\nAggregator]
    AGG --> OUT[Annotated Video\n+ JSON Report]

    OUT --> API
    API -->|GET /api/results| FE

### The Homography Matrix
The matrix $H$ has 8 degrees of freedom:

$$
H = \begin{bmatrix} 
h_{11} & h_{12} & h_{13} \\\\ 
h_{21} & h_{22} & h_{23} \\\\ 
h_{31} & h_{32} & 1 
\end{bmatrix}
$$

* **Top-left $2 \times 2$**: Handles rotation, scaling, and shearing.
* **Third Column**: Handles translation ($x, y$ shifts).
* **Bottom Row**: Handles the perspective warp (making lines parallel again).

## 2. Mathematical Calculations

### Step 1: Solving for $H$
The function `cv2.getPerspectiveTransform` takes 4 pairs of points. Each point provides two linear equations. Since there are 8 unknowns in the matrix, **4 points** are the mathematical minimum required to solve the system.

### Step 2: Point Transformation
To map a worker's detection (e.g., feet at $[x, y]$) to the ground, we use **Homogeneous Coordinates**. We add a third dimension $w=1$:

$$
\begin{bmatrix} x' \\\\ y' \\\\ w' \end{bmatrix} = H \cdot \begin{bmatrix} x \\\\ y \\\\ 1 \end{bmatrix}
$$

### Step 3: Perspective Division (Normalization)
The resulting $x'$ and $y'$ are in "projective space." To get the final pixel coordinates on your 2D map, the computer must divide by the scaling factor $w'$:

$$
\text{Map}_X = \frac{x'}{w'} \quad , \quad \text{Map}_Y = \frac{y'}{w'}
$$

## 3. Function Explanations

### `get_birdseye_view(frame)`
* **Input**: A raw CCTV frame.
* **Process**: Calculates the $H$ matrix and warps the entire image using **Bilinear Interpolation** (`INTER_LINEAR`).
* **Purpose**: Creates the visual "Map" of the factory floor.

### `map_detection_to_ground(coords, matrix)`
* **Input**: $[x, y]$ coordinates (ideally the bottom-center of a YOLO bounding box).
* **Process**: Applies the matrix multiplication and perspective division to that specific point.
* **Efficiency**: This is significantly faster than warping the whole image because it only calculates the transformation for a single pixel coordinate.

## 4. Why Use "Bottom-Center" for Coordinates?
When mapping detections to a ground plane, using the center of a bounding box (the person's waist) will result in a "floating" error. By using the **bottom-center** (the feet), we ensure the coordinate exists exactly on the $Z=0$ plane where the homography matrix is valid.


## Documentation and Articles

| Article | Link |
| :--- | :--- |
| The introduction | https://ayushgarg282800.substack.com/p/what-makes-a-computer-vision-project |
| The research methodology | https://shubhampathneja21.substack.com/p/the-closed-loop-workflow-a-better |
