# AI_media tools documentation
### Video Demo:  <URL HERE>
### Description: 
AI_media is a set of tools to handle multimedia content by taking advantage of functionality provided by Open AI and several open source python libraries. 

These tools are implemented as functions that can be accesed via command line arguments or by importing them into a python script. 

AI_media implements type hints and docstrings accross all functions.

### Functionality

Spotify Plus is a Flask-based web application designed to enhance the Spotify user experience by displaying additional insights about specific tracks and artists. These insights are generated using Open AI's GPT 3.5 Turbo large language model. The app seamlessly interacts with Spotify playlists and devices, and the Open AI model, via real-time API calls. 

To optimize performance and minimize costs associated with OpenAI API usage, Spotify Plus utilizes a SQLite database to store and reuse insights generated by GPT. This caching mechanism ensures that repeated queries for the same song or artist are efficient and cost-effective.

The app is written as a single python file (**app.py**) that uses several packages (e.g. Flask, SQLite3, requests, Open AI).

### Functionality:

#### 1) User authentication
Upon launching Spotify Plus, users are prompted to log in with their Spotify credentials (**index.html**). This initiates the OAuth 2.0 authorization process needed to access Spotify user data. If the user hasn't been authorized before, the app guides the user through a Spotify-hosted screen where the user can grant view/read-only access to the data used by Spotify Plus (i.e. playlists, account data and content being played).

The authorization process generates an access token and a refresh token. The access token is used in all subsequent Spotify API calls. The refresh token is used to automatically get a new access token if/when the access token expires. These tokens are stored along with user name and user_id as session variables. A logout option in **layout.html** allows the user to delete all the data stored in the session, including the tokens. 

#### 2) Display current track

After taking the user through the Spotify authorization process, the app displays the track currently playing in the Spotify app (if any). The user can click a button to get insights about this track. 

#### 3) Display users' playlists and select track

The app also displays a list of all the playlists in the user account (**index.html**) and allows the user to navigate through this list to select a different track from the one being played (**playlist.html**). The app displays a maximum of 10 playlists or tracks in each screen (page), but includes buttons at the bottom of the screen to navigate accross all pages.  

#### 4) Display insights about track and artists

Once the user selects a particular track (either the track being played in Spotify, or one picked from a playlist), the app displays details about the track and all the artists involved (**track.html** ). The details include an image associated with the tracks' album provided by Spotify, and text insights provided by GPT. 

Insights are generated using Chat Completion APIs provided by Open AI. Two separate API calls are done, one for the track, and one for the artists. The prompt given to GPT varies slightly based on the scenario, but include the system message "You are a music erudite and historian, passionate about all music genres." and the request: "Tell me something interesting about"...

#### 5) Display insights about an artist

In the scenario where multiple artists are involved with a track, the insights provided cover all the artists. The user can narrow this by clicking on one particular artist to go to a different page dedicated to that artist alone (**artist.html**). In this page the app generates new insights about the individual artist via a separate call to GPT's Chat Completion API. 

#### 6) Caching of insights from GPT

When users revisit previously explored tracks or artists, Spotify Plus retrieves insights from a local database (**SpotifyPlus.db**) instead of making real-time API calls. This database is continuously updated whenever new insights are fetched from GPT, optimizing performance and minimizing unnecessary API requests.

The database contains two simple tables, one for tracks and one for artists. 

#### 7) Play track, playlist or artist

Accross all pages, wherever a track, playlist or artist is displayed, the user can click on a button to start playing that track, playlist or artist immediately in  Spotify. Two different "play" buttons are provided, one for the Spotify App and one for Spotify Web. The Spotify Web button launches Spotify in a different tab in the browser. 

#### 8) Look and Feel

 The app leverages [Boostrap](https://getbootstrap.com/) for general formatting and navigation (see **layout.html**). Additional CSS definitions are included in **styles.css**


### Endpoints

Spotify Plus supports the following endpoints:

- /: Home page
- /login: Spotify login and authorization
- /logout: Logout and session data deletion
- /callback: Callback for Spotify authorization
- /playlists: Display user playlists
- /playlists/offset=<int:offset>: Paginated playlists
- /playlist/<playlist_id>: Display tracks in a playlist
- /playlist/<playlist_id>/offset=<int:offset>: Paginated tracks in a playlist
- /track/<track_id>: Display insights for a track and all artists involved
- /artist/<artist_id>: Display insights for an artist
- /refresh_token: Refresh Spotify access token

### Technologies and tools used in development

- Python
- Flask
- Bootstrap
- SQLite
- CSS
- HTML
- Git / Github
- VSCode (with CoPilot enabled).

### Setup 

To run Spotify Plus in a local computer, please setup keys to access Spotify and Open AI APIs.

1. Sign up for Spotify and Spotify API ([developer.spotify.com](https://developer.spotify.com/)) if you haven't already and obtain your Spotify Client ID and Spotify Client Secret
2. Sign up for an OPEN AI account ([openai.com](https://openai.com/)) if you haven't already and obtain your OPEN AI API key. 
3. Create a .env file in the application directory and include your Spotify keys:
   - CLIENT_ID = "your client id here"
   - CLIENT_SECRET = "your client secret here"
   - OPENAI_API_KEY = "your Open AI key here"
