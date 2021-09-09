import os
from tensorflow.keras.models import load_model
from flask import (
    Flask,
    request,
    redirect,
    url_for,
    send_from_directory,
    render_template,
)
from werkzeug.utils import secure_filename
from web_app.predict import predict_breed


ALLOWED_EXTENSIONS = set(["png", "jpg", "jpeg", "gif"])
MODEL_FILE = "web_app/static/saved_model/my_model.h5"
STATIC_FOLDER = os.path.join(os.getcwd(), "web_app/static")
UPLOAD_FOLDER = os.path.join(os.getcwd(), "web_app/static/uploads")
BREED_PHOTO_FOLDER = os.path.join(os.getcwd(), "web_app/static/example_images")
FAILED_DETECTION_PIC = "red_x.png"


app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["BREED_FOLDER"] = BREED_PHOTO_FOLDER
classifier = load_model(MODEL_FILE, compile=False)


def allowed_file(filename):
    """Checks if a filename complies with the allowed file endings."""
    return "." in filename and filename.rsplit(".", 1)[1] in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def upload_file():
    """Uploads an image via a html input form."""
    if request.method == "POST":
        file = request.files["file"]
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            filepath = filepath.replace("\\", "/")  # required for windows machine
            file.save(filepath)
            return redirect(url_for("uploaded_file", filename=filename))
    return render_template("master.html")


@app.route("/uploads/<filename>")
def send_file(filename):
    """Sends the file path of the uploaded image to the frontend."""
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route("/show/<filename>")
def uploaded_file(filename):
    """Predicts the dog breed and sends the result to the frontend."""
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    species, breed, breed_dir = predict_breed(filepath, classifier)
    if not species:
        breed_img = [FAILED_DETECTION_PIC]
    else:
        breed_img = [
            s
            for s in os.listdir(app.config["BREED_FOLDER"])
            if breed_dir.split(".")[1] in s
        ]

    return render_template(
        "go.html",
        filename=filename,
        species=species,
        breed=breed,
        breed_img=breed_img[0],
    )


@app.route("/show_breed_photo/<id>")
def send_breed_photo(id):
    """Sends an example image of a dog that belongs to the predicted breed
    class to the frontend."""
    folder = (
        STATIC_FOLDER if id == FAILED_DETECTION_PIC else f'{app.config["BREED_FOLDER"]}'
    )
    # folder = f'{app.config["BREED_FOLDER"]}'
    return send_from_directory(folder, id)


if __name__ == "__main__":
    app.run(debug=True, use_reloader=True, threaded=False)
