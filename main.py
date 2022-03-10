"""
Author: Zejun Gong
Date: 3/Feb/2022
"""
import pickle
from flask import Flask, request, render_template, jsonify
import praw
import os
import matplotlib.pyplot as plt
import pandas as pd
import bert
from transformers import BertTokenizer, BertModel
import requests
import json
import text_analysis


app = Flask(__name__)
"""
==============
Parameters
==============
"""
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# BERT model parameters:
en_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
en_bert_model = BertModel.from_pretrained('bert-base-uncased',output_hidden_states=True)  # Whether the model returns all hidden-states.
cn_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
cn_bert_model = BertModel.from_pretrained('bert-base-chinese',output_hidden_states=True)  # Whether the model returns all hidden-states.

# tisane api
tisane_url = "https://api.tisane.ai/parse"
tisane_headers = {
    'Content-Type': 'application/json',
    'Ocp-Apim-Subscription-Key': 'da1791fc7a024a49a67ff7c7a0d94949'
}


# generate a reddit instance
reddit = praw.Reddit(client_id='3HaNtnWTfosCmcDUHfoyBA',
                     client_secret='yd7zXATTyuu13OLERzTVJF_poYG3qg',
                     user_agent='Political data crawling')


def reddit_query(subreddit_name,choice,num_posts,num_comments):
    subreddit = reddit.subreddit(subreddit_name)
    if choice == 'Top':
        return_dict = {}
        for post in subreddit.top(limit=num_posts):
            comment_arr = []
            _id = post.id
            submission = reddit.submission(id=_id)
            submission_name = submission.title
            # print(submission_name)
            submission.comments.replace_more(limit=None)
            for comment in submission.comments.list()[:num_comments]:  # check the comment forest
                comment = comment.body
                if True:
                    comment = os.linesep.join([s for s in comment.splitlines() if s])  # remove empty lines
                    comment = comment.strip()
                comment_arr.append(comment)
                # print(comment.body)
            return_dict[submission_name] = comment_arr

    elif choice == 'Hot':
        return_dict = {}
        for post in subreddit.hot(limit=num_posts):
            comment_arr = []
            _id = post.id
            try:
                submission = reddit.submission(id=_id)
                submission_name = submission.title
                # print(submission_name)
                submission.comments.replace_more(limit=None)
                for comment in submission.comments.list()[:num_comments]:  # check the comment forest
                    comment = comment.body
                    if True:
                        comment = os.linesep.join([s for s in comment.splitlines() if s])  # remove empty lines
                        comment = comment.strip()
                    comment_arr.append(comment)
                    # print(comment.body)
                return_dict[submission_name] = comment_arr
            except:
                print("exception")
                continue

    elif choice == 'New':
        return_dict = {}
        for post in subreddit.new(limit=num_posts):
            comment_arr = []
            _id = post.id
            try:
                submission = reddit.submission(id=_id)
                submission_name = submission.title
                # print(submission_name)
                submission.comments.replace_more(limit=None)
                for comment in submission.comments.list()[:num_comments]:  # check the comment forest
                    comment = comment.body
                    if True:
                        comment = os.linesep.join([s for s in comment.splitlines() if s])  # remove empty lines
                        comment = comment.strip()
                    comment_arr.append(comment)
                    # print(comment.body)
                return_dict[submission_name] = comment_arr
            except:
                print("exception")
                continue
    print(return_dict)
    return return_dict

@app.route('/')
def load_website():
    return render_template('index.html')

@app.route('/topic_data_crawling',methods=["GET","POST"])
def topic_data_crawling():
    #  once we receive this post request:
    #   1.get the topic name from textarea
    #   2.get the selected option
    #   3.get the user input of Post number/Comment Number
    if request.method == "POST":
        request_data = request.get_json()  # python dictionary
        topic = request_data['topic']  # string
        choice = request_data['choice']
        post_num = request_data['post_num']
        comment_num = request_data['comment_num']
        return_data = reddit_query(topic, choice, int(post_num), int(comment_num))
        # print(return_data)
        return jsonify(return_data)


def pie_chart_drawing(topic,party):
    """
    In debugging mode, the commented code works fine because
    it is fine to dynamically draw the
    pie charts and sent to the front end. However, Matplotlib
    forbids opening a GUI window on the server trying to rendering
    the figure to a png and then shipping it to the user as the payload of a response.
    Thus, the solution is to draw all pie charts before hand and save to static folder.
    """
    # fig = plt.figure()
    # df = pd.read_csv("static/datamodel/all_data.csv")
    # data_arr = []
    # count_df = df.groupby(['topic', 'subreddit']).size().reset_index(name='counts')
    # # print(count_df)
    # try:
    #     democrats_data = count_df[(count_df['topic'] == topic) & (count_df['subreddit'] == 'democrats')].counts.values[0]
    # except:
    #     democrats_data = 0
    # try:
    #     republican_data = count_df[(count_df['topic'] == topic) & (count_df['subreddit'] == 'Republican')].counts.values[0]
    # except:
    #     republican_data = 0
    # try:
    #     conservative_data = count_df[(count_df['topic'] == topic) & (count_df['subreddit'] == 'Conservative')].counts.values[0]
    # except:
    #     conservative_data = 0
    # try:
    #     liberal_data = count_df[(count_df['topic'] == topic) & (count_df['subreddit'] == 'Liberal')].counts.values[0]
    # except:
    #     liberal_data = 0
    # data_arr.append(democrats_data)
    # data_arr.append(republican_data)
    # data_arr.append(conservative_data)
    # data_arr.append(liberal_data)
    # # print(data_arr)
    # labels = ["democrats", "Republican", "Conservative", "Liberal"]
    # explode_index = labels.index(party)
    # explode = [0, 0, 0, 0]
    # explode[explode_index] = 0.2
    # plt.pie(data_arr, labels=labels, explode=explode, shadow=True,autopct='%1.1f%%')
    # plt.axis('equal')
    # plt.show()
    # fig.savefig("static/datamodel/pie_charts/" + topic + '_' + party + '.png',transparent=True)
    return_img_data = {'image_url': ["static/datamodel/pie_charts/" + topic + '_' + party + '.png']}
    return return_img_data
# pie_chart_drawing("/static/datamodel/prochoice","Liberal")

def check_submission(id):
    submission_ids = [id]
    submission_id = [i if i.startswith('t3_') else f't3_{i}' for i in submission_ids]
    for submission in reddit.info(submission_id):
        post_title = submission.title
    return post_title

def check_comment(id):
    comment_instance = reddit.comment(id)
    comment = comment_instance.body
    comment = os.linesep.join([s for s in comment.splitlines() if s])  # remove empty lines
    comment = comment.strip()
    return comment


def cross_checking_data(topic,party):
    return_data = {}
    df = pd.read_csv("static/datamodel/all_data.csv")
    df.set_index(['topic', 'subreddit'], inplace=True)
    df = df.sort_index()
    try:
        kept_df = df.loc[(topic, party)]
    except:
        print("There is no cross records of the " + party + " party")
        return return_data
    kept_df.drop(columns=['author_id', 'author'],inplace=True)
    kept_df.reset_index(drop=True, inplace=True)
    kept_df.set_index(['topic_post_id'], inplace=True)
    # print(kept_df)
    comment_counts = kept_df.groupby(level=[0]).size().reset_index(name='counts')
    cleaned_comment_counts = comment_counts[comment_counts["counts"] >= 3]
    cleaned_index_list = cleaned_comment_counts['topic_post_id'].to_list()
    top3_index_list = cleaned_index_list[:3]
    # print(top3_index_list)
    # get a list of comment ids(only top three inorder to display the comment body)
    if len(top3_index_list) == 3:
        top3_comment_list1 = ((kept_df.loc[top3_index_list[0]])['comment_id'].to_list())[:3]
        top3_comment_list2 = ((kept_df.loc[top3_index_list[1]])['comment_id'].to_list())[:3]
        top3_comment_list3 = ((kept_df.loc[top3_index_list[2]])['comment_id'].to_list())[:3]
        # convert comment ids to comment body
        top3_comment_list1 = [check_comment(i) for i in top3_comment_list1]
        top3_comment_list2 = [check_comment(i) for i in top3_comment_list2]
        top3_comment_list3 = [check_comment(i) for i in top3_comment_list3]
        return_data[top3_index_list[0]] = top3_comment_list1
        return_data[top3_index_list[1]] = top3_comment_list2
        return_data[top3_index_list[2]] = top3_comment_list3
        # rename the old keys(post id) to new keys(post title)
        for i in range(3):
            return_data[check_submission(top3_index_list[i])] = return_data.pop(top3_index_list[i])
        #print(return_data)
    else:
        print("There is no cross records of the " + party + " party")
    return return_data


@app.route('/cross_checking',methods=["GET","POST"])
def cross_checking():
    if request.method == "POST":
        request_data2 = request.get_json()  # python dictionary
        topic = request_data2['section3_topic']  # string
        party = request_data2['party']
        return_img_data = pie_chart_drawing(topic, party)
        return_table_data = cross_checking_data(topic, party)
        return_table_data.update(return_img_data)
        print(return_table_data)
        return return_table_data


@app.route('/prediction',methods=["GET","POST"])
def prediction():
    if request.method == "POST":
        outcome = ''
        request_data3 = request.get_json()  # python dictionary
        political_speech = request_data3['political_speech']  # string
        language_option = request_data3['language']
        if language_option == "English":
            payload = json.dumps({
                "language": "en",
                "content": political_speech,
                "settings": {'abuse': True,
                             'sentiment': True,
                             'entities':True,
                             'topics': True,
                             'explain': True,
                             'parses':True,
                             'snippets':True
                             }
            })
            prediction_return_data = requests.request("POST", tisane_url, headers=tisane_headers, data=payload)

            prediction_json = json.loads(prediction_return_data.text)
            print(json.dumps(prediction_json, indent=4))
            svm_politics_model = pickle.load(open('static/datamodel/en_politics_model.svm', 'rb'))
            word_embedding = bert.generate_sentence_embedding(political_speech, en_tokenizer, en_bert_model)
            probs_list = svm_politics_model.predict_proba(word_embedding)
            republican_probs_value = probs_list[0][0]
            democrats_probs_value = probs_list[0][1]
            if democrats_probs_value > republican_probs_value:
                outcome += 'left-wing comment'
            else:
                outcome += 'right-wing comment'
            #loaded_model = pickle.load(open('static/datamodel/en_hate-speech_model.lr', 'rb'))
            #hate_score = loaded_model.predict(word_embedding)
            hate_score = text_analysis.check_profanity([political_speech])

            if hate_score[0] > 0.5:
                outcome += '/hate speech'
            if hate_score[0] < 0.3:
                severity = "low"
                outcome += '/non-hate'
            elif 0.3 < hate_score[0] < 0.6:
                severity = "medium"
            elif hate_score[0] > 0.6:
                severity = "high"
            else:
                severity = "Cannot detect severity"
            hate_score = str(100* hate_score[0]) + '%'
            #  speech length
            political_speech_length = len(political_speech.split())

            # get names from the political speech
            #people = text_analysis.get_human_names(political_speech)
            people = []
            if not people:
                try:
                    people = [(i["name"] + "| ") for i in prediction_json["entities_summary"] if i["type"] == "person"]
                except:
                    people = "None"

            # get noun phrases from the political speech
            try:
                noun_phrases = [(i["text"] + "| ")for i in prediction_json["abuse"]]
            except:
                noun_phrases = "None"

            filtered_text = text_analysis.filter_profanity(political_speech)
            return_dict = {"democrats":str(100 * democrats_probs_value)+'%',"republicans":str(100 * republican_probs_value)+'%',
                           "hate":hate_score,"outcome":outcome,"word_length":political_speech_length,
                           "people":people,"noun_phrases":noun_phrases,"abuse_severity":severity,
                           "filtered_text":filtered_text}
            print(return_dict)
            return jsonify(return_dict)
        if language_option == 'zh_Chinese':
            print(political_speech)
            payload = json.dumps({
                "language": "zh-CN",
                "content": political_speech,
                "settings": {'abuse': True,
                             'sentiment': True,
                             'topics': True,
                             'entities':True,
                             'explain': True,
                             'parses': True,
                             'snippets': True
                             }
            })
            prediction_return_data = requests.request("POST", tisane_url, headers=tisane_headers, data=payload)
            prediction_json = json.loads(prediction_return_data.text)
            print(json.dumps(prediction_json, indent=4))
            cn_svm_politics_model = pickle.load(open('static/datamodel/cn_politics_model.lr', 'rb'))
            word_embedding = bert.generate_sentence_embedding(political_speech, cn_tokenizer, cn_bert_model)
            probs_list = cn_svm_politics_model.predict_proba(word_embedding)
            pro_beijing_probs_value = probs_list[0][0] # pro_beijing is labeled as 0
            anti_beijing_probs_value = probs_list[0][1]

            if pro_beijing_probs_value > anti_beijing_probs_value:
                outcome += 'left-wing comment'
            else:
                outcome += 'right-wing comment'
            cn_loaded_model = pickle.load(open('static/datamodel/cn_hate-speech_model.svm', 'rb'))
            cn_hate_prob_list = cn_loaded_model.predict_proba(word_embedding)
            print(cn_hate_prob_list)
            hate_score = cn_hate_prob_list[0][1]  # sexiest is labeled as 1

            if hate_score > 0.5:
                outcome += '/hate speech'
            if hate_score < 0.3:
                severity = "low"
                outcome += '/non-hate'
            elif 0.3 < hate_score < 0.6:
                severity = "medium"
            elif hate_score > 0.6:
                severity = "high"
            else:
                severity = "Cannot detect severity"
            political_speech_length = text_analysis.hans_count(political_speech)

            # get names from the political speech(Chinese)
            #people = text_analysis.cn_get_human_names(political_speech)
            people = []
            if not people:
                try:
                    people = [(i["name"] + "| ") for i in prediction_json["entities_summary"] if i["type"] == "person"]
                    if not people:
                        people = "None"
                except:
                    people = "None"
            # get noun phrases from the political speech
            try:
                noun_phrases = [(i["text"] + "| ") for i in prediction_json["abuse"]]
            except:
                noun_phrases = "None"

            return_dict = {"democrats": str(100 * pro_beijing_probs_value) + '%', "republicans": str(100 * anti_beijing_probs_value) + '%',
                           "hate": str(100 * hate_score) + '%' , "outcome": outcome, "word_length": political_speech_length,
                           "people": people, "noun_phrases": noun_phrases, "abuse_severity": severity}

            print(return_dict)
            return jsonify(return_dict)


if __name__ == '__main__':
    app.run(host='127.0.0.1',port=8080,debug=True)