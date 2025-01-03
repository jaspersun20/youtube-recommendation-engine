import requests

# TMDb and OMDb API keys
tmdb_api_key = "b8792d13412713da445d0d478ea34150"
omdb_api_key = "6145e2e1"

# Example IMDb and TMDb IDs for movies 1 and 2
movies = [
    {"imdb_id": "tt0114709", "tmdb_id": 862},  # Movie ID 1

]


# Function to fetch movie details from TMDb
def fetch_tmdb_movie(tmdb_id):
    tmdb_url = f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={tmdb_api_key}"
    response = requests.get(tmdb_url)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"TMDb API Error: {response.status_code}")
        return None


# Function to fetch movie details from OMDb (IMDb data)
def fetch_imdb_movie(imdb_id):
    omdb_url = f"http://www.omdbapi.com/?i={imdb_id}&apikey={omdb_api_key}"
    response = requests.get(omdb_url)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"OMDb API Error: {response.status_code}")
        return None


# Fetch details for each movie
for movie in movies:
    print(f"Fetching data for IMDb ID: {movie['imdb_id']}, TMDb ID: {movie['tmdb_id']}\n")

    # Fetch TMDb data
    tmdb_data = fetch_tmdb_movie(movie["tmdb_id"])
    if tmdb_data:
        print("TMDb Data:")
        print(tmdb_data)

    # Fetch IMDb data
    imdb_data = fetch_imdb_movie(movie["imdb_id"])
    if imdb_data:
        print("IMDb (OMDb) Data:")
        print(imdb_data)

    print("\n" + "-" * 50 + "\n")
