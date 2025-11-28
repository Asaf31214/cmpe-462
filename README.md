### We have a data pipeline of notebooks, that can be run to obtain the usable tabular data from raw images, and train the models with it.

### Dependencies
- If you have the `uv` package manager, running `uv sync` at root directory is enough to install the necessary libraries.
- Otherwise, you need to manually install `matplotlib, numpy, opencv-python, polars, scikit-learn, seaborn` with your preferred package manager.

### Dataset
Publicly accessible [here](https://drive.google.com/drive/folders/1Cceit5q3lhH0x4lQarAn-ayfdtF1-35G)

## Steps
1. Place the image data (generated ones and photographs) into data/image folder like this:
```
data/image:
├── generated
│   ├── banana
│   │   └── *.png
│   ├── carrot
│   │   └── *.png
│   ├── cucumber
│   │   └── *.png
│   ├── mandarin
│   │   └── *.png
│   └── tomato
│       └── *.png
└── photograph
    ├── banana
    │   └── *.png
    ├── carrot
    │   └── *.png
    ├── cucumber
    │   └── *.png
    ├── mandarin
    │   └── *.png
    └── tomato
        └── *.png
```
2. Run the notebook `image_processing.ipynb`. This notebook splits the data into training, validation, and test and resizes the images to standard 512x512.
3. Set the desired feature types (`image, numeric, text`) at 4th cell of the `feature_extraction.ipynb` notebook and run it. For the full fused data, all should be set to `True`.
4. Run the `preprocessing.ipynb` notebook. This notebook does the multi-hot encoding for the description column using the bag of words method.
5. Run the `logistic_regression.ipynb` notebook. This is our implementation of a multi-class logistic regression using the one vs. all approach. Might take few minutes to complete.
6. (Optional) Run the `sklearn-logistic.ipynb` notebook if you want to compare the performance and metrics to Scikit Learn implementation on same data.

### Similarity Anaylsis
Running the `similarity_analysis.ipynb` notebook shows the intra-class and inter-class similarity analysis and the outlier detection.

### Extra
There is also the `image_generation.ipynb` notebook which is not a part of this standard pipeline. It is used to generate synthetic image. 

To run it, you need to load the notebook in Google Colab, or any other environment with a GPU. 
- Tune the `batch_size` according to the VRAM of the environment (25 works fine for A100 with 40 GB VRAM).
- Add your HuggingFace token at the environment secrets. This step is required to access models like `stabilityai/stable-diffusion-3.5-medium` which are not fully public.
- Mount your Google Drive storage, so that the generated images do not get lost if the session ends. By default, images are saved into the folder `462-images`, directory is created if not already exists. 
- Run the notebook. With batch size of 25 and A100 GPU, an average of 50 images/min is generated.

