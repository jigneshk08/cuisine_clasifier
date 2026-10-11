# Cuisine Classification

Predicts a restaurant's primary cuisine from city, price, rating, votes, services and name.

## Folder layout
cuisine_classifier/
  cuisine_classifier.py   <- the project
  Dataset_.csv            <- the data (must sit next to the script)
  requirements.txt
  outputs/                <- created automatically (charts + CSV tables)

## Run (VS Code terminal)
python -m venv venv
venv\Scripts\activate          (Windows)   |   source venv/bin/activate   (Mac/Linux)
pip install -r requirements.txt
python cuisine_classifier.py
