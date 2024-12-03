from flask import Flask, render_template, request
import random

app = Flask(__name__)
search_query = ""  # The query from the search bar is stored here in a simple string


# TODO: Outsource this function into a IR model in another file
# Generate random search results
def generate_search_results(count=10):
    return [{
        "author": f"Author {random.randint(1, 100)}",
        "title": f"Title {random.randint(1, 100)}",
        "link": "https://example.com"
    } for _ in range(count)]


# Functionality of the home page
@app.route("/", methods=["GET", "POST"])
def home():
    global search_query
    search_query = ""       # Default to empty
    selected_iir = "Name"  # Default value
    border_color = "red"    # Default border color
    search_results = []     # Default empty search results

    if request.method == "POST":
        search_query = request.form.get("search")   # The content of the search bar
        selected_iir = request.form.get("option")  # The selected IR method (currently not implemented 02.12.2024)

        # Generate search results
        search_results = generate_search_results()  # TODO: use the correct IR method for this

        # Sets border color based on selected option
        # TODO: option should be a IR method
        if selected_iir == "Name":
            border_color = "red"
        elif selected_iir == "Age":
            border_color = "blue"
        elif selected_iir == "Misc":
            border_color = "green"

        # Used for debugging purposes
        print(f"Search query received: {search_query}, Option selected: {selected_iir}")

    # supply the parameters used by the html file
    return render_template(
        "index.html",
        title="MMSR - Project",
        search_results=search_results,
        selected_option=selected_iir,
        border_color=border_color,
        search_query=search_query,
    )


if __name__ == "__main__":
    app.run(debug=True)
