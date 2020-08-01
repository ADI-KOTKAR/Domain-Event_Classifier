# -*- coding: utf-8 -*-

def classifyDomain(file):
    import pandas as pd
    df = pd.read_csv('Dataset/Domains_Dataset2.csv')
    df.head()
    
    from io import StringIO
    col = ['Domain', 'Event']
    df = df[col]
    df = df[pd.notnull(df['Event'])]
    df.columns = ['Domain', 'Event']
    df.head()
        
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
    features = tfidf.fit_transform(df.Event).toarray()
    labels = df.Domain
    print(features.shape)
        
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.svm import LinearSVC
    X_train, X_test, y_train, y_test = train_test_split(df['Event'], df['Domain'], random_state = 0)
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(X_train)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    clf = LinearSVC().fit(X_train_tfidf, y_train)
        
    import csv
    typelist = []
    with open('{}'.format(file),'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for line in csv_reader:
            this_dict = {}
            #print(line)
            #print(clf.predict(count_vect.transform([str(line)])))
            this_dict['event'] = str(line)
            this_dict['domain'] = str(clf.predict(count_vect.transform([str(line)])))
            typelist.append(this_dict)
            del this_dict
            #print(classifyArray)
        return typelist
    

def classifyType(file):
    import pandas as pd
    df = pd.read_csv('Dataset/Events_Dataset.csv')
    df.head()
    
    from io import StringIO
    col = ['Type', 'Event']
    df = df[col]
    df = df[pd.notnull(df['Event'])]
    df.columns = ['Type', 'Event']
    df.head()
        
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
    features = tfidf.fit_transform(df.Event).toarray()
    labels = df.Type
    #print(features.shape)
        
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.svm import LinearSVC
    X_train, X_test, y_train, y_test = train_test_split(df['Event'], df['Type'], random_state = 0)
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(X_train)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    clf2 = LinearSVC().fit(X_train_tfidf, y_train)
        
    import csv
    
    hi = classifyDomain(file)
    
    with open('{}'.format(file),'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for line,i in zip(csv_reader, range(0, len(hi))):
            #print(line)
            #print(clf2.predict(count_vect.transform([str(line)])))
            prediction = str(clf2.predict(count_vect.transform([str(line)])))
            #print(prediction)
            hi[i]['type'] = prediction
            hi[i]['emp'] = []
        return hi
        

  


