from flask import Flask, render_template, request, jsonify, session
import pandas as pd
import json
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALSModel
from pyspark.sql.functions import col, explode, desc, split, array_contains, regexp_replace, lower
import os
from datetime import datetime
import logging
import requests
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this in production

class MovieRecommendationApp:
    # Class-level variables to store model and data
    _model = None
    _movies_df = None
    _genres_list = []
    _is_data_loaded = False

    def __init__(self):
        self.spark = None
        self.is_initialized = False
        # TMDB API configuration
        self.tmdb_api_key = os.getenv('TMDB_API_KEY')  # Get API key from environment variable
        self.tmdb_base_url = "https://api.themoviedb.org/3"
        self.tmdb_image_base = "https://image.tmdb.org/t/p/w500"
        # Initialize Spark session immediately
        self.initialize_spark()

    @classmethod
    def load_model_and_data(cls, spark_session):
        """Load model and data only once"""
        if cls._is_data_loaded:
            return True

        try:
            logger.info("Loading ALS model...")
            model_path = "/mnt/d/python/project/movie_recommendation_model"
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model path {model_path} does not exist")
            
            cls._model = ALSModel.load(model_path)

            logger.info("Loading movies data...")
            movies_path = "/mnt/d/python/project/preprocessed_data/movies_data"
            if not os.path.exists(movies_path):
                raise FileNotFoundError(f"Movies data path {movies_path} does not exist")
            
            cls._movies_df = spark_session.read.parquet(movies_path)
            cls._movies_df.cache()
            cls._movies_df.count()  # Trigger cache population

            # Extract unique genres
            cls._extract_genres()

            cls._is_data_loaded = True
            logger.info("Model and data loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Error loading model and data: {e}")
            return False

    @classmethod
    def _extract_genres(cls):
        """Extract unique genres from the movies dataset"""
        try:
            # Get all genres from the dataset
            genres_df = cls._movies_df.select(
                explode(split(col("genres"), "\\|")).alias("genre")
            ).distinct().filter(col("genre") != "(no genres listed)")

            cls._genres_list = [row['genre'] for row in genres_df.collect()]
            cls._genres_list.sort()
            logger.info(f"Extracted {len(cls._genres_list)} unique genres")

        except Exception as e:
            logger.error(f"Error extracting genres: {e}")
            # Fallback to common genres
            cls._genres_list = [
                "Action", "Adventure", "Animation", "Children", "Comedy",
                "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
                "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
                "Thriller", "War", "Western"
            ]

    def initialize_spark(self):
        try:
            print("Initializing Spark session...")
            logger.info("Initializing Spark session...")
            
            # Set Hadoop home and disable native libraries
            os.environ['HADOOP_HOME'] = os.path.join(os.getcwd(), 'hadoop')
            os.environ['HADOOP_OPTS'] = '-Djava.library.path={}'.format(
                os.path.join(os.getcwd(), 'hadoop', 'bin')
            )
            
            # Only create a new Spark session if one doesn't exist
            if self.spark is None:
                self.spark = SparkSession.builder \
                    .appName("MovieRecommendationApp") \
                    .master("local[*]") \
                    .config("spark.driver.memory", "12g") \
                    .config("spark.executor.memory", "8g") \
                    .config("spark.sql.adaptive.enabled", "true") \
                    .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
                    .config("spark.sql.shuffle.partitions","4") \
                    .config("spark.hadoop.io.native.lib.available", "false") \
                    .config("spark.driver.host", "localhost") \
                    .getOrCreate()

            # Load model and data if not already loaded
            if not self.__class__._is_data_loaded:
                success = self.__class__.load_model_and_data(self.spark)
                if not success:
                    raise Exception("Failed to load model and data")

            self.is_initialized = True
            logger.info("Spark session initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Error initializing Spark: {e}")
            self.is_initialized = False
            return False

    def get_movie_poster_url(self, title, movie_id):
        """Get movie poster URL from TMDB API"""
        if not self.tmdb_api_key:
            logger.warning("TMDB API key not set. Using placeholder images.")
            return self._get_placeholder_poster(title, movie_id)

        try:
            # Clean the title for better search results
            clean_title = title.split('(')[0].strip()
            
            # Search for the movie
            search_url = f"{self.tmdb_base_url}/search/movie"
            params = {
                'api_key': self.tmdb_api_key,
                'query': clean_title,
                'language': 'en-US',
                'page': 1,
                'include_adult': False
            }
            
            response = requests.get(search_url, params=params, timeout=5)
            response.raise_for_status()
            
            data = response.json()
            if data['results']:
                # Get the first result
                movie = data['results'][0]
                poster_path = movie.get('poster_path')
                
                if poster_path:
                    return f"{self.tmdb_image_base}{poster_path}"
                
            # If no poster found, use placeholder
            return self._get_placeholder_poster(title, movie_id)
            
        except Exception as e:
            logger.warning(f"Error fetching TMDB poster for {title}: {e}")
            return self._get_placeholder_poster(title, movie_id)

    def _get_placeholder_poster(self, title, movie_id):
        """Generate a placeholder poster when TMDB API fails or is not configured"""
        colors = ['FF6B6B', '4ECDC4', '45B7D1', 'FFA07A', '98D8C8', 'F7DC6F', 'BB8FCE', '85C1E9']
        color = colors[movie_id % len(colors)]
        title_encoded = title.replace(' ', '+').replace('(', '').replace(')', '')[:20]
        return f"https://via.placeholder.com/300x450/{color}/FFFFFF?text={title_encoded}"

    def enhance_movie_data(self, movies_list):
        """Add poster URLs to movie data"""
        for movie in movies_list:
            movie['poster_url'] = self.get_movie_poster_url(movie['title'], movie['movieId'])
            #print(f"movie['title'] : {movie['title']}")
            # Extract year from title if available
            if '(' in movie['title'] and ')' in movie['title']:
                year_match = movie['title'].split('(')[-1].split(')')[0]
                if year_match.isdigit():
                    movie['year'] = year_match
                    movie['clean_title'] = movie['title'].split('(')[0].strip()
                else:
                    movie['year'] = 'N/A'
                    movie['clean_title'] = movie['title']
            else:
                movie['year'] = 'N/A'
                movie['clean_title'] = movie['title']
            
            # Ensure title is not too long for display
            if len(movie['clean_title']) > 40:
                movie['clean_title'] = movie['clean_title'][:37] + '...'
            
            # Ensure we have a title even if it's missing
            if not movie['clean_title']:
                movie['clean_title'] = 'Untitled Movie'
            

        
        return movies_list

    def is_spark_running(self):
        """Check if Spark session is running"""
        try:
            return (self.spark is not None and 
                    hasattr(self.spark, '_jsc') and 
                    not self.spark._jsc.sc().isStopped())
        except:
            return False

    def get_user_recommendations(self, user_id, num_recs=10):
        if not self.is_initialized or not self.is_spark_running():
            logger.error("Spark session is not initialized or stopped.")
            return []
        try:
            user_df = self.spark.createDataFrame([(int(user_id),)], ["userId"])
            user_recs = self.__class__._model.recommendForUserSubset(user_df, num_recs)

            if user_recs.count() == 0:
                return []

            recs_exploded = user_recs.select(
                col("userId"),
                explode(col("recommendations")).alias("recommendation")
            ).select(
                col("userId"),
                col("recommendation.movieId").alias("movieId"),
                col("recommendation.rating").alias("predicted_rating")
            )

            recs_with_info = recs_exploded.join(self.__class__._movies_df, "movieId") \
                                         .select("movieId", "title", "genres", "predicted_rating") \
                                         .orderBy(desc("predicted_rating"))

            movies_list = recs_with_info.toPandas().to_dict('records')
            return self.enhance_movie_data(movies_list)

        except Exception as e:
            logger.error(f"Error getting user recommendations: {e}")
            return []

    def search_movies(self, query, limit=20):
        if not self.is_initialized or not self.is_spark_running():
            logger.error("Spark session is not initialized or stopped.")
            return []

        try:
            search_results = self.__class__._movies_df.filter(
                col("title").contains(query)
            ).limit(limit)

            movies_list = search_results.toPandas().to_dict('records')
            return self.enhance_movie_data(movies_list)

        except Exception as e:
            logger.error(f"Error searching movies: {e}")
            return []

    def get_movies_by_genre(self, genre, limit=20, sort_by="title"):
        """Get movies filtered by genre"""
        if not self.is_initialized or not self.is_spark_running():
            logger.error("Spark session is not initialized or stopped.")
            return []

        try:
            # Filter movies that contain the specified genre
            genre_movies = self.__class__._movies_df.filter(
                col("genres").contains(genre)
            )

            # Sort by specified criteria
            if sort_by == "title":
                genre_movies = genre_movies.orderBy(col("title"))
            elif sort_by == "year":
                # Extract year from title and sort by it
                genre_movies = genre_movies.orderBy(desc(
                    regexp_replace(col("title"), r".*\((\d{4})\).*", "$1")
                ))

            # Limit results
            genre_movies = genre_movies.limit(limit)

            movies_list = genre_movies.toPandas().to_dict('records')
            return self.enhance_movie_data(movies_list)

        except Exception as e:
            logger.error(f"Error getting movies by genre: {e}")
            return []

    def get_genres_list(self):
        """Return list of available genres"""
        return self.__class__._genres_list

    def get_genre_statistics(self):
        """Get statistics about movies per genre"""
        if not self.is_initialized or not self.is_spark_running():
            logger.error("Spark session is not initialized or stopped.")
            return {}

        try:
            genre_stats = {}
            for genre in self.__class__._genres_list:
                count = self.__class__._movies_df.filter(col("genres").contains(genre)).count()
                genre_stats[genre] = count

            # Sort by count descending
            genre_stats = dict(sorted(genre_stats.items(), key=lambda x: x[1], reverse=True))
            return genre_stats

        except Exception as e:
            logger.error(f"Error getting genre statistics: {e}")
            return {}

    def get_movie_details(self, movie_id):
        """Get detailed movie information including TMDB data"""
        if not self.is_initialized or not self.is_spark_running():
            logger.error("Spark session is not initialized or stopped.")
            return None

        try:
            movie_info = self.__class__._movies_df.filter(col("movieId") == int(movie_id))

            if movie_info.count() > 0:
                movie_dict = movie_info.toPandas().iloc[0].to_dict()
                
                # If TMDB API is configured, get additional details
                if self.tmdb_api_key:
                    try:
                        # Search for the movie in TMDB
                        search_url = f"{self.tmdb_base_url}/search/movie"
                        params = {
                            'api_key': self.tmdb_api_key,
                            'query': movie_dict['title'].split('(')[0].strip(),
                            'language': 'en-US',
                            'page': 1,
                            'include_adult': False
                        }
                        
                        response = requests.get(search_url, params=params, timeout=5)
                        response.raise_for_status()
                        
                        data = response.json()
                        if data['results']:
                            tmdb_movie = data['results'][0]
                            # Add TMDB data to movie dict
                            movie_dict.update({
                                'tmdb_id': tmdb_movie.get('id'),
                                'overview': tmdb_movie.get('overview'),
                                'release_date': tmdb_movie.get('release_date'),
                                'vote_average': tmdb_movie.get('vote_average'),
                                'vote_count': tmdb_movie.get('vote_count'),
                                'popularity': tmdb_movie.get('popularity')
                            })
                    except Exception as e:
                        logger.warning(f"Error fetching TMDB details for movie {movie_id}: {e}")
                
                # Enhance with poster URL
                enhanced_movies = self.enhance_movie_data([movie_dict])
                return enhanced_movies[0] if enhanced_movies else movie_dict
            return None

        except Exception as e:
            logger.error(f"Error getting movie details: {e}")
            return None

    def get_popular_movies(self, limit=20):
        if not self.is_initialized or not self.is_spark_running():
            logger.error("Spark session is not initialized or stopped.")
            return []

        try:
            popular_movies = self.__class__._movies_df.limit(limit)
            
            movies_list = popular_movies.toPandas().to_dict('records')
            
            return self.enhance_movie_data(movies_list)

        except Exception as e:
            logger.error(f"Error getting popular movies: {e}")
            return []

    def get_similar_movies_by_genre(self, movie_id, limit=10):
        """Get movies similar to a given movie based on shared genres"""
        if not self.is_initialized or not self.is_spark_running():
            logger.error("Spark session is not initialized or stopped.")
            return []

        try:
            # Get the target movie's genres
            target_movie = self.__class__._movies_df.filter(col("movieId") == int(movie_id)).collect()
            if not target_movie:
                return []

            target_genres = target_movie[0]['genres'].split('|')

            # Find movies with similar genres (excluding the target movie)
            similar_movies = self.__class__._movies_df.filter(col("movieId") != int(movie_id))

            # Filter movies that share at least one genre
            for genre in target_genres:
                if genre != "(no genres listed)":
                    similar_movies = similar_movies.filter(col("genres").contains(genre))
                    break  # Use first valid genre for initial filter

            similar_movies = similar_movies.limit(limit)
            movies_list = similar_movies.toPandas().to_dict('records')
            return self.enhance_movie_data(movies_list)

        except Exception as e:
            logger.error(f"Error getting similar movies: {e}")
            return []

    def cleanup(self):
        """Clean up Spark resources"""
        if self.spark:
            try:
                self.spark.stop()
                self.spark = None
                self.is_initialized = False
                logger.info("Spark session stopped successfully")
            except Exception as e:
                logger.error(f"Error stopping Spark session: {e}")


# Initialize the recommendation system
rec_app = None

def initialize_app():
    """Initialize the recommendation app with proper error handling"""
    global rec_app
    if rec_app is None:
        rec_app = MovieRecommendationApp()
        success = rec_app.initialize_spark()
        if not success:
            logger.error("Failed to initialize recommendation system")
            return False
    return rec_app.is_initialized

# Initialize on import
with app.app_context():
    initialize_app()

@app.route('/')
def home():
    if not rec_app or not rec_app.is_initialized:
        return render_template('500.html', error="Recommendation system not initialized."), 500
    
    try:
        popular_movies = rec_app.get_popular_movies(12)
        print(f"Popular Movies: {popular_movies}")
        genres_list = rec_app.get_genres_list()
        print("Rendering Index template")
        return render_template('index.html', popular_movies=popular_movies, genres_list=genres_list)
    except Exception as e:
        logger.error(f"Error in home route: {e}")
        return render_template('500.html', error=str(e)), 500

@app.route('/search')
def search():
    if not rec_app or not rec_app.is_initialized:
        return render_template('500.html', error="Recommendation system not initialized."), 500
    
    try:
        query = request.args.get('q', '')
        movies = []
        if query:
            movies = rec_app.search_movies(query)
        return render_template('search.html', query=query, movies=movies)
    except Exception as e:
        logger.error(f"Error in search route: {e}")
        return render_template('500.html', error=str(e)), 500

@app.route('/movie/<int:movie_id>')
def movie_details(movie_id):
    if not rec_app or not rec_app.is_initialized:
        return render_template('500.html', error="Recommendation system not initialized."), 500
    
    try:
        movie = rec_app.get_movie_details(movie_id)
        if not movie:
            return render_template('404.html'), 404

        # Get similar movies
        similar_movies = rec_app.get_similar_movies_by_genre(movie_id, 8)
        return render_template('movie_details.html', movie=movie, similar_movies=similar_movies)
    except Exception as e:
        logger.error(f"Error in movie details route: {e}")
        return render_template('500.html', error=str(e)), 500

@app.route('/recommendations')
def recommendations():
    if not rec_app or not rec_app.is_initialized:
        return render_template('500.html', error="Recommendation system not initialized."), 500
    
    try:
        # Get user ID from query parameters, default to 1 if not provided
        user_id = request.args.get('user_id', 1, type=int)
        
        # Get recommendations for the user
        recommendations = rec_app.get_user_recommendations(user_id, num_recs=10)
        
        return render_template('recommendations.html', 
                             recommendations=recommendations,
                             user_id=user_id)
    except Exception as e:
        logger.error(f"Error in recommendations route: {e}")
        return render_template('500.html', error=str(e)), 500

@app.route('/genres')
def genres():
    if not rec_app or not rec_app.is_initialized:
        return render_template('500.html', error="Recommendation system not initialized."), 500
    
    try:
        genres_list = rec_app.get_genres_list()
        genre_stats = rec_app.get_genre_statistics()
        return render_template('genres.html', genres=genres_list, genre_stats=genre_stats)
    except Exception as e:
        logger.error(f"Error in genres route: {e}")
        return render_template('500.html', error=str(e)), 500

@app.route('/genre/<genre_name>')
def genre_movies(genre_name):
    if not rec_app or not rec_app.is_initialized:
        return render_template('500.html', error="Recommendation system not initialized."), 500
    
    try:
        sort_by = request.args.get('sort', 'title')
        limit = request.args.get('limit', 20, type=int)

        movies = rec_app.get_movies_by_genre(genre_name, limit, sort_by)
        genre_stats = rec_app.get_genre_statistics()

        return render_template('genre_movies.html',
                             genre=genre_name,
                             movies=movies,
                             sort_by=sort_by,
                             total_count=genre_stats.get(genre_name, 0))
    except Exception as e:
        logger.error(f"Error in genre movies route: {e}")
        return render_template('500.html', error=str(e)), 500

# API Routes
@app.route('/api/recommendations/<int:user_id>')
def api_get_recommendations(user_id):
    if not rec_app or not rec_app.is_initialized:
        return jsonify({'error': 'Recommendation system not initialized'}), 500
    
    try:
        num_recs = request.args.get('limit', 10, type=int)
        recommendations = rec_app.get_user_recommendations(user_id, num_recs)
        return jsonify({
            'user_id': user_id,
            'recommendations': recommendations,
            'count': len(recommendations)
        })
    except Exception as e:
        logger.error(f"Error in API recommendations: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/search')
def api_search():
    if not rec_app or not rec_app.is_initialized:
        return jsonify({'error': 'Recommendation system not initialized'}), 500
    
    try:
        query = request.args.get('q', '')
        limit = request.args.get('limit', 20, type=int)

        if not query:
            return jsonify({'error': 'Query parameter is required'}), 400

        movies = rec_app.search_movies(query, limit)
        return jsonify({
            'query': query,
            'movies': movies,
            'count': len(movies)
        })
    except Exception as e:
        logger.error(f"Error in API search: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/movie/<int:movie_id>')
def api_movie_details(movie_id):
    if not rec_app or not rec_app.is_initialized:
        return jsonify({'error': 'Recommendation system not initialized'}), 500
    
    try:
        movie = rec_app.get_movie_details(movie_id)
        if not movie:
            return jsonify({'error': 'Movie not found'}), 404
        return jsonify(movie)
    except Exception as e:
        logger.error(f"Error in API movie details: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/popular')
def api_popular_movies():
    if not rec_app or not rec_app.is_initialized:
        return jsonify({'error': 'Recommendation system not initialized'}), 500
    
    try:
        limit = request.args.get('limit', 20, type=int)
        movies = rec_app.get_popular_movies(limit)
        return jsonify({
            'movies': movies,
            'count': len(movies)
        })
    except Exception as e:
        logger.error(f"Error in API popular movies: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/genres')
def api_genres():
    if not rec_app or not rec_app.is_initialized:
        return jsonify({'error': 'Recommendation system not initialized'}), 500
    
    try:
        genres = rec_app.get_genres_list()
        genre_stats = rec_app.get_genre_statistics()
        return jsonify({
            'genres': genres,
            'statistics': genre_stats
        })
    except Exception as e:
        logger.error(f"Error in API genres: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/genre/<genre_name>')
def api_genre_movies(genre_name):
    if not rec_app or not rec_app.is_initialized:
        return jsonify({'error': 'Recommendation system not initialized'}), 500
    
    try:
        sort_by = request.args.get('sort', 'title')
        limit = request.args.get('limit', 20, type=int)

        movies = rec_app.get_movies_by_genre(genre_name, limit, sort_by)
        return jsonify({
            'genre': genre_name,
            'movies': movies,
            'count': len(movies),
            'sort_by': sort_by
        })
    except Exception as e:
        logger.error(f"Error in API genre movies: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/similar/<int:movie_id>')
def api_similar_movies(movie_id):
    if not rec_app or not rec_app.is_initialized:
        return jsonify({'error': 'Recommendation system not initialized'}), 500
    
    try:
        limit = request.args.get('limit', 10, type=int)
        similar_movies = rec_app.get_similar_movies_by_genre(movie_id, limit)
        return jsonify({
            'movie_id': movie_id,
            'similar_movies': similar_movies,
            'count': len(similar_movies)
        })
    except Exception as e:
        logger.error(f"Error in API similar movies: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/spark-ui')
def spark_ui():
    if not rec_app or not rec_app.is_initialized:
        return "Spark is not running.", 500
    ui_url = rec_app.spark.sparkContext.uiWebUrl
    if ui_url:
        # Redirect or provide a clickable link
        return f'<a href="{ui_url}" target="_blank">Open Spark UI</a>'
    else:
        return "Spark UI is not available.", 500

@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return render_template('500.html', error="Internal server error occurred"), 500

# Health check endpoint
@app.route('/health')
def health_check():
    status = {
        'status': 'healthy' if rec_app and rec_app.is_initialized else 'unhealthy',
        'spark_running': rec_app.is_spark_running() if rec_app else False,
        'timestamp': datetime.now().isoformat()
    }
    return jsonify(status)

if __name__ == '__main__':
    try:
        # Ensure initialization is complete
        if not initialize_app():
            print("Failed to initialize the recommendation system. Exiting.")
            exit(1)
        
        # Run the Flask application
        app.run(host='0.0.0.0', port=5000, debug=True)
        
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
        if rec_app:
            rec_app.cleanup()
    except Exception as e:
        print(f"Error starting server: {e}")
        logger.error(f"Server startup error: {e}")
        if rec_app:
            rec_app.cleanup() 