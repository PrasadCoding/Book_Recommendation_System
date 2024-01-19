from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

popu_df = pickle.load(open('pkl_files/top.pkl', 'rb'))
pt = pickle.load(open('pkl_files/pt_new.pkl', 'rb'))
user_fam_df = pickle.load(open('pkl_files/user_fam_df.pkl', 'rb'))
sim_score = pickle.load(open('pkl_files/sim_score.pkl', 'rb'))
df = pickle.load(open('pkl_files/df.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/popularity/')
def popularity():
    return render_template('popularity.html',
                           book_title = list(popu_df['Book-Title'].values),
                           book_auth = list(popu_df['Book-Author'].values),
                           book_pub = list(popu_df['Year-Of-Publication'].values),
                           book_img = list(popu_df['Image-URL-L'].values), 
                           book_rating = list(popu_df['Avg-Rating'].values),
                           )

@app.route('/collaborative/')
def collaborative():
    return render_template('collaborative.html')

@app.route('/content/')
def content():
    return render_template('content.html')

@app.route('/recommend_books', methods=["GET", "POST"])
def collab_recommend():
    try: 
        user_input = request.form.get('bname')
        book_ind = np.where(pt.index == user_input)[0]
        if len(book_ind) == 0:
            return render_template('collaborative.html')
        book_ind = book_ind[0]
        book_sim_index = enumerate(sim_score[book_ind])
        book_sim_sorted = sorted(book_sim_index, key=lambda x:x[1], reverse = True)[1:6]
        book_title_list = []
        book_author_list = []
        book_year_list = []
        book_image_list = []
        for i in book_sim_sorted:
            print(pt.index[i[0]])
            book_info = user_fam_df[user_fam_df['Book-Title'] == pt.index[i[0]]].iloc[:1]
            book_title_list.append(book_info['Book-Title'].values[0])
            book_author_list.append(book_info['Book-Author'].values[0])
            book_year_list.append(book_info['Year-Of-Publication'].values[0])
            book_image_list.append(book_info['Image-URL-L'].values[0]) 

        return render_template('collaborative.html',
                            book_title = book_title_list,
                            book_auth = book_author_list,
                            book_pub = book_year_list,
                            book_img = book_image_list, 
                            )

    except:
        pass

@app.route('/content_recommend', methods=["GET", "POST"])
def content_recommend():
    try:
        content_user_input = request.form.get('content_b_name')
        data = df.loc[df['title'] == content_user_input]
        
        if len(data) == 0:
            return render_template('content.html')
        
        data = df.loc[df['genre'] == data['genre'][0]]  
        data.reset_index(level = 0, inplace = True) 
        indices = pd.Series(data.index, index = data['title'])
        tf = TfidfVectorizer(analyzer='word', ngram_range=(2, 2), min_df = 1, stop_words='english')
        tfidf_matrix = tf.fit_transform(data['title'])
        # Calculating the similarity measures based on Cosine Similarity
        sg = cosine_similarity(tfidf_matrix, tfidf_matrix)
        idx = indices[content_user_input]
        sig = list(enumerate(sg[idx]))
        sig = sorted(sig, key=lambda x: x[1], reverse=True)[1:6]
        title_list = []
        author_list = []
        genre_list = []
        image_list = []
        for i in sig:
            title_list.append(data['title'][i[0]])
            author_list.append(data['author'][i[0]])
            genre_list.append(data['genre'][i[0]])
            image_list.append(data['image_link'][i[0]])
        
        return render_template('content.html',
                                title = title_list,
                                author = author_list,
                                genre = genre_list,
                                image = image_list, 
                                )
    except:
        pass
    

if __name__ == '__main__':
    app.run(debug=True)


