# Plake (Iraqi Plate Detection)

## Overview

This project aims to develop an automated system for detecting and blurring license plates in car images, specifically focusing on Iraqi vehicle plates. The project involves data collection, annotation, and model training using state-of-the-art techniques to achieve reliable plate detection.

## Table of Contents

- [Plake (Iraqi Plate Detection)](#plake-iraqi-plate-detection)
  - [Overview](#overview)
  - [Table of Contents](#table-of-contents)
  - [Data Collection](#data-collection)
  - [Annotation](#annotation)
  - [Modeling](#modeling)
  - [Installation](#installation)
  - [Contributing](#contributing)
  - [License](#license)

## Data Collection

- **Description**: 80,000 Images were crawled and stored in drive
- **Preprocessing**: A zero-shot image classification model("openai/clip-vit-base-patch32") was employed to classify images into two categories: "has plate" and "does not have plate."

## Annotation

- **Tool**: The annotation of images was carried out using the Roboflow platform.
- **Dataset Size**: Over 1,500 images were manually annotated, identifying the license plates' locations within the images.

## Modeling

- **Model Used**: The YOLOv8-OBB (Oriented Bounding Box) model was selected for fine-tuning on the annotated dataset.
- **Training**: The model was trained and fine-tuned to detect license plates in various orientations and lighting conditions.
- **Feature**: The trained model is capable of detecting license plates in images and can automatically blur the detected plates to ensure privacy.

## Installation

To replicate the project, follow these steps:

1. Clone this repository:

    ```bash
    git clone https://github.com/Mmli081/plate.git
    cd plate
    ```

2. Create a virtual environment and install the required packages:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

## Contributing

Contributions to improve the project are welcome. You can start by forking this repository, making your changes, and submitting a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
