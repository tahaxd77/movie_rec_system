
---

# Movie Recommendation System

A web-based Movie Recommendation System built with Python and Flask, leveraging machine learning techniques to provide personalized movie suggestions. This project includes a content-based recommendation engine and a user-friendly interface for exploring movies by genre, searching titles, and viewing detailed movie information.

## Table of Contents

- [About the Project](#about-the-project)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)


## About the Project

This Movie Recommendation System aims to help users discover movies tailored to their interests. It uses content-based filtering techniques implemented in Python (see `recommendation_system.ipynb`) and exposes the functionality through a Flask web application (`app.py`). The app allows users to browse movies by genre, search for movies, view detailed information, and receive personalized recommendations.

## Features

- Content-based movie recommendations using machine learning
- Browse movies by genres
- Search movies by title
- View detailed movie information
- Responsive web interface with HTML templates
- Error handling with custom 404 and 500 pages


## Technologies Used

- Python
- Flask (Web Framework)
- Jupyter Notebook (for model development)
- HTML / CSS (Frontend templates)
- Machine Learning libraries (e.g., scikit-learn, pandas)


## Installation

1. Clone the repository:

```bash
git clone https://github.com/tahaxd77/movie_recommendation_system.git
cd movie_recommendation_system
```

2. Create and activate a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

*(If `requirements.txt` is not present, install Flask and other dependencies manually, e.g., `pip install flask pandas scikit-learn`)*

## Usage

1. Run the Flask application:

```bash
python app.py
```

2. Open your web browser and go to:

```
http://127.0.0.1:5000/
```

3. Use the interface to browse genres, search for movies, and get recommendations.

## Project Structure

```
movie_recommendation_system/
│
├── app.py                    # Flask application entry point
├── recommendation_system.ipynb  # Jupyter notebook with recommendation model development
├── templates/                # HTML templates for rendering web pages
│   ├── base.html
│   ├── index.html
│   ├── genres.html
│   ├── genre_movies.html
│   ├── movie_details.html
│   ├── recommendations.html
│   ├── search.html
│   ├── 404.html
│   └── 500.html
├── static/                   # Static files (CSS, JS, images) if any
└── README.md                 # Project documentation
```


## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

---
