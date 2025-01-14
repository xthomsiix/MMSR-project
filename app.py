from enum import Enum
import logging
from typing import Any, Dict, Hashable, List
from flask import Flask, render_template, request
from services.mmrs import MultiMediaRetrievalSystem
from services.data import DatasetLoader
from services.common import IRMethod

# Configure the main logger for the application
logging.basicConfig(
    level=logging.WARNING,  # Set to DEBUG or another level as needed
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

dataset_loader: DatasetLoader = DatasetLoader()
dataset_loader.load()
mmrs: MultiMediaRetrievalSystem = MultiMediaRetrievalSystem()
mmrs.prepare_data(
    dataset_loader.id_artist_song_album,
    dataset_loader.id_url,
    dataset_loader.id_genres,
    dataset_loader.id_metadata,
    dataset_loader.id_tags,
)


class BorderColor(Enum):
    RED = "red"
    BLUE = "blue"
    GREEN = "green"
    ORANGE = "orange"
    PURPLE = "purple"
    MAROON = "maroon"
    BROWN = "brown"
    YELLOW = "yellow"


# initialize logger
logger: logging.Logger = logging.getLogger(__name__)

# initialize Flask app
app = Flask(__name__)


# Functionality of the home page
@app.route("/", methods=["GET", "POST"])
def home():
    # initialize global variables for text fields with default values
    artist: str | None = request.form.get("artist")
    song_title: str | None = request.form.get("songTitle")
    N: int = int(request.form.get("numResults", 10))
    selected_iir = request.form.get("option", IRMethod.BASELINE.value)
    border_color: str | None = BorderColor.RED.value
    ir_results: Dict[str, str | float | List[Dict[Hashable, Any]] | None] = {}

    if request.method == "POST" and artist and song_title:
        # make a switch case for the different IIR methods
        match selected_iir:
            case IRMethod.BASELINE.value:
                border_color = BorderColor.RED.value
                ir_results = mmrs.baseline(artist, song_title, N)
            case IRMethod.TFIDF.value:
                border_color = BorderColor.GREEN.value
                ir_results = mmrs.tfidf(dataset_loader.tfidf, artist, song_title, N)
            case IRMethod.BERT.value:
                border_color = BorderColor.BLUE.value
                ir_results = mmrs.bert(dataset_loader.bert, artist, song_title, N)
            case IRMethod.BLF_SPECTRAL.value:
                border_color = BorderColor.ORANGE.value
                ir_results = mmrs.blf_spectral(
                    dataset_loader.blf_spectral, artist, song_title, N
                )
            case IRMethod.MUSIC_NN.value:
                border_color = BorderColor.PURPLE.value
                ir_results = mmrs.music_nn(
                    dataset_loader.music_nn, artist, song_title, N
                )
            case IRMethod.RESNET.value:
                border_color = BorderColor.MAROON.value
                ir_results = mmrs.resnet(dataset_loader.resnet, artist, song_title, N)
            case IRMethod.VGG19.value:
                border_color = BorderColor.BROWN.value
                ir_results = mmrs.vgg19(dataset_loader.vgg19, artist, song_title, N)
            case IRMethod.LLM.value:
                border_color = BorderColor.YELLOW.value
                ir_results = mmrs.llm(dataset_loader.llm, artist, song_title, N)
            case _:  # default case not implemented
                raise NotImplementedError("Default method not implemented yet")

        logger.debug(
            f"Selected IIR method: {selected_iir}, border color: {border_color}"
        )
        logger.info(f"Search query received: artist: {artist} - title: {song_title}")

    # supply the parameters used by the html file
    return render_template(
        # Metadata for the HTML page
        "index.html",
        title="MMSR - Project",
        # Data for the search form
        search_results=ir_results.get("search_results", []),
        selected_option=selected_iir,
        # rendering parameters
        border_color=border_color,
        # query-specific data
        artist=artist,
        songTitle=song_title,
        precision=ir_results.get("precision"),
        recall=ir_results.get("recall"),
        ndcg=ir_results.get("ndcg"),
        mrr=ir_results.get("mrr"),
        message=ir_results.get("message"),
        songs_by_artist=mmrs.autocomplete(),
    )


if __name__ == "__main__":
    app.run(debug=True)
