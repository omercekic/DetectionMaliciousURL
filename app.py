from urllib.parse import urlparse

import numpy as np
import streamlit as st
import pandas as pd
from tld import get_tld

from prediction import predict
import feature_extraction

st.title("Welcome to Malicious URL Website Detection App!")
st.markdown("Find out if the website you want to access is safe.")
# st.text("Enter the URL of the website you want to access 👇")
url = st.text_input("Enter the URL of the website 👇",
                    label_visibility="visible",
                    disabled=False,
                    placeholder="URL here!", )

button = st.button("Detect URL!")


def URL_Converter(url):
    data = pd.DataFrame()
    data['url'] = pd.Series(url)

    data['url'] = data['url'].replace('www.', '', regex=True)

    data['url_len'] = data['url'].apply(lambda x: len(str(x)))

    data['entropy'] = data['url'].apply(lambda i: feature_extraction.entropy(i))

    data['count-www'] = data['url'].apply(lambda i: i.count('www'))

    feature = ['@', '?', '-', '=', '.', '%', '//', '/']
    for a in feature:
        data[a] = data['url'].apply(lambda i: i.count(a))

    data['abnormal_url'] = data['url'].apply(lambda i: feature_extraction.abnormal_url(i))

    data['https'] = data['url'].apply(lambda i: feature_extraction.httpSecure(i))

    data['fd_length'] = data['url'].apply(lambda i: feature_extraction.fd_length(i))

    data['tld'] = data['url'].apply(lambda i: get_tld(i, fail_silently=True))

    data['tld_length'] = data['tld'].apply(lambda i: feature_extraction.tld_length(i))

    data['digits'] = data['url'].apply(lambda i: feature_extraction.digit_count(i))

    data['letters'] = data['url'].apply(lambda i: feature_extraction.letter_count(i))

    data['sus_url'] = data['url'].apply(lambda i: feature_extraction.suspicious_words(i))

    data['hostname_length'] = data['url'].apply(lambda i: len(urlparse(i).netloc))

    data['shortining_service'] = data['url'].apply(lambda x: feature_extraction.shortining_service(x))

    data['having_ip_address'] = data['url'].apply(lambda i: feature_extraction.having_ip_address(i))

    X = data.drop(['url', 'tld'], axis=1)

    return X

if button:
    # result = predict(np.array([[url]]))
    # st.text(np.array([url]))

    # print(np.array([url]))
    # user_url = np.array([url])
    #
    # user_url = user_url.replace('www.', '', regex=True)
    # print(user_url)

    user_url = URL_Converter(url)
    result = predict(user_url)

    if (np.array_equal(result,[0])):
        answer = "This website is Legitimate! You can safely access. :)"
    elif (np.array_equal(result, [1])):
        answer = "This website is defacement!"
    elif (np.array_equal(result, [2])):
        answer = "This website can be included phishing! Please be careful!!!"
    elif (np.array_equal(result,[3])):
        answer = "This website is malicious! Access is not recommended!"

    st.text(answer)

