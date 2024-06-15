from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

import tensorflow as tf
import os
import matplotlib.pyplot as plt
from numpy import *
from tensorflow.keras.models import load_model
import tensorflow_hub as hub

import numpy as np
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import easyocr


from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from spellchecker import SpellChecker

import warnings
warnings.filterwarnings("ignore")

app = Flask('__name__')


path_to_upload =None
global_text=""
count=0
lang=None


# Load the models with the custom_objects parameter
custom_objects = {
    'KerasLayer': hub.KerasLayer
}

model = load_model('detector.keras', custom_objects=custom_objects)


@app.route('/', methods = ['GET', 'POST'])
def intro_start():
    return render_template("intro.html") 



@app.route('/langauge_detect', methods = ['GET', 'POST'])
def langauge():
    if(request.method == 'POST'):
        f=request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        global path_to_upload

        path_to_upload=file_path
        
        print(file_path)


        # Make prediction
       
        img=plt.imread(file_path)
        

        #resize according to layers
        resize=tf.image.resize(img,(256,256))
        resize=resize/255
        #expand your image array
        img=expand_dims(resize,0)
        predictions=model.predict(img)

        

        # Convert the predicted probabilities to class labels
        predicted_labels = np.argmax(predictions, axis=1)

    

        ref={0:"English",1:"Hindi",2:"Telugu"}

        res=ref[predicted_labels[0]]

        global lang

        if(res=="English"):
            lang='en'
        elif(res=="Hindi"):
            lang='hi'
        elif(res=="Telugu"):
            lang='te'
        else:
            lang=None
            
    return render_template("detect.html",n=res)



def extract_ocr():

    global lang

    if(lang!=None):
        # Set the logging level to ERROR or CRITICAL to suppress warnings and info messages
        logging.getLogger().setLevel(logging.ERROR)
        

        # Initialize the EasyOCR reader with the desired language(s)
        reader = easyocr.Reader([lang])  # You can specify multiple languages if needed
        
        # Load your image
        image_path = path_to_upload

        # Perform OCR on the image
        results = reader.readtext(image_path)

        # Print the extracted text
        global count
        
        count=0
        final_text=""
        for (bbox, text, prob) in results:
            final_text=final_text+" "+text
            count=count+len(text)

        global global_text
        global_text=final_text

        #once text is extracted lang in None making avalible for next process
        #lang=None
        
        return final_text

    else:
        return "Language is not Identified"
    

@app.route('/extraction', methods = ['GET', 'POST'])
def extract_text_language():

    res=extract_ocr()

    if(res=="Language is not Identified"):
        return render_template("error.html",n=res)

    return render_template("extracted.html",n=res)


def gpt_format(value):
    
    # Load pre-trained model and tokenizer
    model_name = "gpt2"  # You can use a different pre-trained model
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Input text
    input_text = value

    # Tokenize the input text
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    global count
    print(count)

    # Generate corrected text
    corrected_ids = model.generate(input_ids, max_length=count, num_return_sequences=1, no_repeat_ngram_size=1,attention_mask=torch.ones(input_ids.shape),  # Set attention mask
                               pad_token_id=tokenizer.eos_token_id)

    # Decode and print the corrected text
    corrected_text = tokenizer.decode(corrected_ids[0], skip_special_tokens=True)
    #print("Original Text:", input_text)
    #print("Corrected Text:", corrected_text)
    
    return corrected_text



def normal_format(value):
    spell = SpellChecker()
    # Find and correct misspelled words in a text
    text = value
    words = text.split()
    corrected_words = [spell.correction(word) for word in words]
    corrected_text = " ".join(corrected_words)

    return corrected_text




def format_text(text, line_length=80):
    # Split the text into lines based on line_length
    lines = []
    current_line = ""
    for word in text.split():
        # Remove unwanted symbols like $, @, _
        #word = ''.join(char for char in word if char not in ('$@_'))

        if len(current_line) + len(word) + 1 <= line_length:
            if current_line:
                current_line += " " + word
            else:
                current_line = word
        else:
            lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)

    # Align text to the left
    formatted_lines = [line.ljust(line_length) for line in lines]

    # Join the lines to create the formatted text
    formatted_text = "\n".join(formatted_lines)

    return formatted_text


@app.route('/gpt', methods = ['GET', 'POST'])
def extract_text_gpt():

    global lang
    
    if(lang=='hi' or lang=='te'):
        return render_template("hindi_telugu.html")

    else:
    
        global global_text

        global_text=gpt_format(global_text)

        
        input_text = global_text
        formatted_text = format_text(input_text)
        #print(formatted_text)

        global_text=formatted_text

        #make lang as NOne for further updates
        #lang=None


        return render_template("end.html")

@app.route('/normal', methods = ['GET', 'POST'])
def extract_text_normal():

    global lang
    global global_text

    if(lang=='hi' or lang=="te"):
        return render_template("hindi_telugu.html")

    else:
        global_text=normal_format(global_text)

        
        input_text = global_text
        formatted_text = format_text(input_text)
        #print(formatted_text)

        global_text=formatted_text


        return render_template("end.html")


def save_text_to_pdf(text, output_file="output.pdf"):

    
    # Create a PDF document
    doc = SimpleDocTemplate(output_file, pagesize=letter)

    # Create a list of flowable objects (e.g., paragraphs)
    story = []

    # Create a paragraph style
    styles = getSampleStyleSheet()
    style = styles["Normal"]

    # Create a paragraph with the text
    paragraph = Paragraph(text, style)
    story.append(paragraph)

    # Build the PDF document
    doc.build(story)



def save_text_to_text_file(text, output_file="output.txt"):
    # Open a text file for writing
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(text)



@app.route('/pdf', methods = ['GET', 'POST'])
def pdf():
    global lang
    input_text = global_text
    save_text_to_pdf(input_text, output_file="output.pdf")

    return render_template("close.html")

@app.route('/text', methods = ['GET', 'POST'])
def text():
    input_text = global_text
    save_text_to_text_file(input_text, output_file="output.txt")

    return render_template("close.html")
    
    
if __name__ == "__main__":
    app.run(debug = False)
