# Word2Vec in Moroccan Dialect

![Status](https://img.shields.io/badge/status-work%20in%20progress-yellow)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/framework-PyTorch-ee4c2c)
![NLP](https://img.shields.io/badge/domain-NLP-green)

> A Word2Vec implementation for Moroccan Darija that enables basic word completion using a custom-trained model built with Python and PyTorch.

---

## About

This project applies the **Word2Vec architecture** to **Moroccan Darija** (Moroccan Arabic dialect), a low-resource language with very limited NLP tooling.

Built as a **practice project during my Data Science studies**, the goal is to train word embeddings that capture semantic relationships in Darija and use them for **word completion tasks**.

⚠️ **Work in Progress**  
1. This project is still under development. Some features may be incomplete and the project structure may change in future updates.
2. To execute  the code , use `run.py`
---

## Features

- Word2Vec model trained on Moroccan Darija text
- Word completion based on learned embeddings
- Built from scratch using PyTorch

---

## Installation & Usage

### Prerequisites

- Python 3.10+
- pip

### Install dependencies

```bash
git clone https://github.com/Ibrahimibrahimi/word2vec-moroccan-dialect.git
cd word2vec-moroccan-dialect
pip install -r requirements.txt
```

### Train the model

```bash
python run.py
```

### Run word completion

```bash
python predict.py --word "your_input_word"
```

---

## Project Structure

This is the planned structure of the project:

```
word2vec-moroccan-dialect/
│
├── data/               # Training corpus
├── model/              # Saved model weights
├── train.py            # Training script
├── predict.py          # Word completion inference
├── requirements.txt
└── README.md
```

---

## Built With

- [Python](https://www.python.org/)
- [PyTorch](https://pytorch.org/)

---

## Roadmap

- [ ] Collect and clean a larger Darija corpus
- [ ] Improve tokenization for Darija-specific patterns
- [ ] Add evaluation metrics (similarity and analogy tasks)
- [ ] Build a simple demo interface

---

## Author

**Ibrahim Id-Wahman**  
Data Science enthusiast exploring NLP and deep learning through hands-on projects.

- GitHub: https://github.com/Ibrahimibrahimi

---

*This project currently has no license. All rights reserved.*
