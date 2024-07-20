#Import libs
import os
from flask import Flask, redirect, render_template, request, url_for, jsonify, session
from utils import image_pipeline, text_pipeline, groq_interface
from pinecone import Pinecone
from dotenv import load_dotenv
load_dotenv()

#Configure the Pinecone API key
pc = Pinecone(os.environ['PINECONE_API_KEY'])
#Setting up the index
index = pc.Index('embeddings')

#Create an instance of the Flask class
app = Flask(__name__)

#Setting the secret key
app.secret_key = os.environ['SECRET_KEY']

#Function to get the response
def get_response(text_input):
    groq_ans = groq_interface.get_groq_respoonse(text_input)
    #Check if there are '~' signs in the beginning and end of the groq_ans
    if groq_ans[0] == '~':
        groq_ans = groq_ans[1:]
    if groq_ans[-1] == '~':
        groq_ans = groq_ans[:-1]
    #Now since the groq ans contains the product with the ~ signs in between
    groq_ans = groq_ans.split('~')  #Splitting the products

    #Dense embeddings of the products
    dense_embeddings = list()
    for product in groq_ans:
        dense_embeddings.append(text_pipeline.get_dense_embeddings(product))
    
    #Sparse embeddings of the products
    sparse_embeddings = list()
    for product in groq_ans:
        sparse_embeddings.append(text_pipeline.get_sparse_embeddings(product))
    
    #Now, we get the vectors in a format to pass on to the Pinecone index
    final_embeddings = list()
    for i in range(len(groq_ans)):
        final_embeddings.append({
            "hdense" : dense_embeddings[i],
            "hsparse" : sparse_embeddings[i]
        })
    
    #Now we pass the final embeddings to the Pinecone index
    pinecone_res = index.query(
        top_k=5,
        namespace='embeds',
        vector=final_embeddings[0]['hdense'],
        query_vector=final_embeddings[0]['hsparse'],
        include_metadata=True
    ) 

    sructured_response = list()
    for response in pinecone_res['matches']:
        sructured_response.append({
            "id": response['id'],
            "name": response['metadata']['name'],
            "description": response['metadata']['description'],
            "price": response['metadata']['price'],
            "image": response['metadata']['image_path'],
            "color" : response['metadata']['color'],
            "category" : response['metadata']['category']
        })
    final_format = {
        'products' : sructured_response
    }
    
    return final_format 




@app.route('/', methods = ['POST', 'GET'])
def api():
    if request.method == 'POST':
        
        #Now, I need to obtain the text from the actual website
        text_input = request.get_json()
        text_input = text_input.get('text', None)
        session['text_input'] = text_input

        return redirect(url_for('results'))  
    
    if request.method == 'GET':
        return render_template('index.html')

@app.route('/results', methods=['GET'])
def results():
    # Just render the template without passing the answer
    return render_template('results.html')

@app.route('/results/data', methods=['GET'])
def results_data():
    response = str(session.get('text_input', None))
    answer = get_response(response)
    return jsonify(answer)



#Run the app
if __name__ == '__main__':
    app.run(debug=True)