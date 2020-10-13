import os
from flask import Flask, flash, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename

from Transforming_audio_data import predict


UPLOAD_FOLDER = './input'
ALLOWED_EXTENSIONS = {'wav'}


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            to = filename
            return redirect(to)
    return '''
<!DOCTYPE html>
<html lang=“en”>
<center>
<head>
    <meta charset=“UTF-8">
    <meta name=“viewport” content=“width=device-width, initial-scale=1.0">
    <title>Neural network tells accents apart</title>
</head>
<body style="background-color:#A5C8E4;">
<h1 style="font-family:verdana">What instrument is playing? 🎺</h1>
</p>
<h2> Upload an audio file so the program can try to guess:</h2>
<form method=post enctype=multipart/form-data>
      <input type=file name=file style="height:50px; width:90px">
      <input type=submit value=⬆️UPLOAD style="height:50px; width:90px">
    </form>
<div class="tenor-gif-embed" data-postid="16768803" data-share-method="host" data-width="50%" data-aspect-ratio="1.7978339350180503"><a href="https://tenor.com/view/duck-playing-djembe-djembe-drum-musical-instrument-duck-gif-16768803">Duck Playing Djembe Drum GIF</a> from <a href="https://tenor.com/search/duckplayingdjembe-gifs">Duckplayingdjembe GIFs</a></div><script type="text/javascript" async src="https://tenor.com/embed.js"></script>
</body>
<center>
</html>
'''

@app.route('/<audioname>', methods=['GET'])
def accent (audioname):
    path = f"input/{audioname}"
    result = predict(path)
    return '''
<!DOCTYPE html>
<html lang=“en”>
<center>
<head>
    <meta charset=“UTF-8">
    <meta name=“viewport” content=“width=device-width, initial-scale=1.0">
    <title>Neural network tells which instrument is playing</title>
</head>
<body style="background-color:#A5C8E4;">
<h1 style="font-family:verdana">What instrument is playing? 🎻</h1>
<div class="tenor-gif-embed" data-postid="5330954" data-share-method="host" data-width="50%" data-aspect-ratio="1.7785714285714287"><a href="https://tenor.com/view/aristocats-berlios-piano-tattletale-grumpy-gif-5330954">Aristocats Berlios GIF</a> from <a href="https://tenor.com/search/aristocats-gifs">Aristocats GIFs</a></div><script type="text/javascript" async src="https://tenor.com/embed.js"></script><form>
    <input type="button" value="Go back" onclick="history.back()">
</form> 
</body>
<center>
</html>
'''.format(result=result)

app.run('0.0.0.0', port=3000, debug=True, threaded=False) #http://0.0.0.0:3000/