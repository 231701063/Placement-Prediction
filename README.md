# PlacementVision AI

PlacementVision AI is a machine learning mini-project that trains a placement prediction model in Python using the provided student placement dataset and shows the results in a polished website.

## Features

- Trains a logistic regression model directly from `placementdata.csv`
- Uses a complete Python backend with a built-in HTTP server
- Shows accuracy, precision, recall, F1 score, and confusion matrix
- Highlights the most influential placement features
- Lets users enter a student profile and get a live placement prediction
- Uses a responsive, presentation-ready website

## Project structure

- `server.py`: Runs the web server and API routes
- `data_loader.py`: Loads the dataset and converts CSV rows into ML features
- `ml_model.py`: Contains scaling, logistic regression training, and evaluation code
- `model_service.py`: Connects dataset loading, model training, metrics, and prediction
- `config.py`: Stores dataset path, public folder path, feature list, and port

## Easy access to the model

If you want to quickly show your ML code during a demo or viva, open these files:

- `ml_model.py` for the actual model training code
- `data_loader.py` for dataset loading and preprocessing
- `model_service.py` for model creation and prediction flow

Important functions:

- `load_dataset()` in `data_loader.py`
- `build_matrix()` in `data_loader.py`
- `train_logistic_regression()` in `ml_model.py`
- `create_model_bundle()` in `model_service.py`
- `predict_placement()` in `model_service.py`

## Run the project locally

1. Open a terminal in this folder.
2. Run `python server.py`
3. Open `http://localhost:3000`

You can also run `npm start` if Python is available in your PATH.

## Host on GitHub Pages

This project can be hosted on GitHub Pages as a fully static website.

1. Run `python export_static_bundle.py`
2. Commit the generated `public/data/model-bundle.json`
3. Push the project to a GitHub repository
4. In GitHub, open `Settings -> Pages`
5. Set the source to `GitHub Actions`
6. Push again after any model or dataset change to redeploy

The live GitHub Pages site uses the exported model bundle in the browser, so it does not need the Python server for hosting.
