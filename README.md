# MNIST CNN – Handwritten Digit Recognition

A handwritten digit recognition application built using **Convolutional Neural Networks (CNNs), PyTorch, and Streamlit**. The application allows users to upload handwritten digit images and obtain predictions through a trained deep learning model.

## Project Overview

The MNIST dataset contains handwritten digits from 0 to 9. This project uses a CNN model to learn patterns from handwritten digits and classify them into their corresponding numerical labels.

The project also provides an interactive Streamlit web interface for image uploading and digit recognition.

## Features

- Handwritten digit recognition using CNN.
- Deep learning model implemented with PyTorch.
- Interactive Streamlit web interface.
- Upload handwritten digit images (PNG, JPG, JPEG).
- Support for MNIST IDX image datasets.
- Display predicted digit and prediction confidence.
- Visualize uploaded handwritten images.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| PyTorch | Deep learning framework |
| Torchvision | Dataset loading and image transformations |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Pillow | Image processing |
| Streamlit | Web application interface |

## Dataset

**MNIST (Modified National Institute of Standards and Technology)**

- 60,000 training images.
- 10,000 testing images.
- 28 × 28 grayscale images.
- 10 classes (digits 0–9).

The project supports loading MNIST IDX image files.

## Project Structure

```text
MNIST/
│
├── app.py
├── model.py
├── requirements.txt
├── README.md
├── trained_model.pth
└── .gitignore
```

*The structure above is illustrative. Update the filenames to match your actual repository.*

## Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Navigate to the project directory

```bash
cd MNIST
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

## How to Use

1. Launch the Streamlit application.
2. Upload a handwritten digit image or an MNIST IDX image dataset.
3. If you upload an MNIST dataset, select an image using the image selector.
4. Click the **Recognize Digit** button.
5. View the predicted digit and confidence score.

## Model

The project uses a Convolutional Neural Network (CNN) for handwritten digit classification.

CNNs are suitable for image classification because they learn spatial features such as edges, shapes, and patterns from images.

## Future Improvements

- Improve model accuracy through hyperparameter tuning.
- Add prediction probability visualizations.
- Support additional image formats.
- Deploy the application online.
- Experiment with deeper CNN architectures.

## Author

**Nikita Kumari**

Computer Science Undergraduate

GitHub: [Your GitHub Profile](https://github.com/YOUR_USERNAME)

---

If you find this project useful, consider giving the repository a ⭐.
