import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from urllib.parse import urlparse
from tld import get_tld
from collections import Counter
import feature_extraction

data = pd.read_csv(r"data/malicious_phish.csv")

data = data.drop_duplicates()


data_edit = {"Category": {"benign": 0, "defacement": 1, "phishing": 2, "malware": 3}}
data['Category'] = data['type']
data = data.replace(data_edit)

# -------Feature Extraction-----------
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

X = data.drop(['type', 'Category', 'url', 'tld'], axis=1)
y = data['Category']

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=101)

print(f"Training target statistics: {Counter(Y_train)}")
print(f"Testing target statistics: {Counter(Y_test)}")

models = [RandomForestClassifier]

for m in models:
    print('#############################################')
    print('######-Model =>\033[07m {} \033[0m'.format(m))
    model_ = m()
    model_.fit(X_train, Y_train)
    pred = model_.predict(X_test)
    acc = accuracy_score(pred, Y_test)
    print('Test Accuracy :\033[32m \033[01m {:.2f}% \033[30m \033[0m'.format(acc * 100))
    print('\033[01m              Classification_report \033[0m')
    print(classification_report(Y_test, pred))
    print('\033[01m             Confusion_matrix \033[0m')
    cf_matrix = pd.DataFrame(confusion_matrix(Y_test, pred),
                             columns=['Predicted:0', 'Predicted:1', 'Predicted:2', 'Predicted:3'],
                             index=['Actual:0', 'Actual:1', 'Actual:2', 'Actual:3'])
    sns.heatmap(cf_matrix, annot=True, fmt=".5g")
    plt.show()
    print('\033[31m###################- End -###################\033[0m')

model = joblib.dump(model_, "rf_model.sav")
