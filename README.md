# A Lightweight Crowd Model for Robot Social Navigation

This repository contains the implementation accompanying the paper:  

**A Lightweight Crowd Model for Robot Social Navigation**  
*Maryam Kazemi Eskeri, Thomas Wiedemann, Ville Kyrki, Dominik Baumann, Tomasz Piotr Kucner*  
Accepted at *IEEE Proceedings, 2025*.

---

## Overview
In this work, we propose a **lightweight macroscopic crowd prediction model** that:  
- Reduces inference time **3.6×** compared to a high-dimensional baseline.  
- Improves prediction accuracy by **3.1%**.  
- Integrates with socially aware planners for real-time robot navigation.  

The model encodes pedestrian density, velocity, and variance on a grid, and forecasts spatiotemporal dynamics using a **ConvRNN-based encoder-forecaster**. Predictions are then integrated into a **PRM\*** planning framework for socially compliant navigation.

---

## Citation

If you use this code, please cite our paper.
