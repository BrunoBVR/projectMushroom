import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree

sns.set(font_scale=1.5)

st.set_option('deprecation.showPyplotGlobalUse', False)

st.markdown('''
# <img style="float: left;" src="https://mario.wiki.gallery/images/thumb/a/a6/Super_Mushroom_Artwork_-_Super_Mario_3D_World.png/1200px-Super_Mushroom_Artwork_-_Super_Mario_3D_World.png" width = 60>   The [mushroom dataset](https://archive.ics.uci.edu/ml/datasets/Mushroom) <img style="float: right;" src="https://i.pinimg.com/originals/a8/ff/3e/a8ff3ed1011dbabc869ab8ea401ace4e.png" width=80>
''', unsafe_allow_html=True)

df_mush = pd.read_csv('../mushrooms.csv')

selected_view = st.selectbox(
    'Would you like to explore the data or do some modeling?',
    ('Let\'s explore!', 'Model away!')
)

if selected_view == 'Let\'s explore!':

    with st.beta_expander("Show intro?"):
        st.markdown('''
    The [mushroom data set](https://archive.ics.uci.edu/ml/datasets/Mushroom) has been contributed to the UCI Machine Learning over 30 years ago. From the authors description:

    > This dataset includes descriptions of hypothetical samples corresponding to 23 species of gilled mushrooms in the Agaricus and Lepiota Family Mushroom drawn from The Audubon Society Field Guide to North American Mushrooms (1981). Each species is identified as definitely edible, definitely poisonous, or of unknown edibility and not recommended. This latter class was combined with the poisonous one. **The Guide clearly states that there is no simple rule for determining the edibility of a mushroom; no rule like "leaflets three, let it be'' for Poisonous Oak and Ivy.**

    Each mushroom is characterized by 22 distinct features:

    * cap-shape: bell=b,conical=c,convex=x,flat=f, knobbed=k,sunken=s
    * cap-surface: fibrous=f,grooves=g,scaly=y,smooth=s
    * cap-color: brown=n,buff=b,cinnamon=c,gray=g,green=r,pink=p,purple=u,red=e,white=w,yellow=y
    * bruises: bruises=t,no=f
    * odor: almond=a,anise=l,creosote=c,fishy=y,foul=f,musty=m,none=n,pungent=p,spicy=s
    * gill-attachment: attached=a,descending=d,free=f,notched=n
    * gill-spacing: close=c,crowded=w,distant=d
    * gill-size: broad=b,narrow=n
    * gill-color: black=k,brown=n,buff=b,chocolate=h,gray=g, green=r,orange=o,pink=p,purple=u,red=e,white=w,yellow=y
    * stalk-shape: enlarging=e,tapering=t
    * stalk-root: bulbous=b,club=c,cup=u,equal=e,rhizomorphs=z,rooted=r,missing=?
    * stalk-surface-above-ring: fibrous=f,scaly=y,silky=k,smooth=s
    * stalk-surface-below-ring: fibrous=f,scaly=y,silky=k,smooth=s
    * stalk-color-above-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y
    * stalk-color-below-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y
    * veil-type: partial=p,universal=u
    * veil-color: brown=n,orange=o,white=w,yellow=y
    * ring-number: none=n,one=o,two=t
    * ring-type: cobwebby=c,evanescent=e,flaring=f,large=l,none=n,pendant=p,sheathing=s,zone=z
    * spore-print-color: black=k,brown=n,buff=b,chocolate=h,green=r,orange=o,purple=u,white=w,yellow=y
    * population: abundant=a,clustered=c,numerous=n,scattered=s,several=v,solitary=y
    * habitat: grasses=g,leaves=l,meadows=m,paths=p,urban=u,waste=w,woods=d

    All features are categorical (and not numeric), so it poses a good exercise for encoding.

    <img style="float: right;" src="https://media.springernature.com/lw785/springer-static/image/chp%3A10.1007%2F978-3-319-17900-1_120/MediaObjects/327013_2_En_120_Fig1_HTML.gif">

    With a quick look at these features, we can notice 5 different kinds of features relating to the "anatomy" of the mushrooms: **cap, gill, stalk, veil and ring**. From the figure we can get an ideia of what each feature is.

    Besides this features, we have **odor**, which has a clear meaning as does **bruises** (you can check for an interesting article on why *magic mushrooms* turn blue when bruised [here](https://www.nature.com/articles/d41586-019-03614-0)). Also, we have **spore-print**, which, according to the [Wikipedia](https://en.wikipedia.org/wiki/Spore_print#:~:text=The%20spore%20print%20is%20the,fall%20onto%20a%20surface%20underneath.&text=It%20shows%20the%20color%20of%20the%20mushroom%20spores%20if%20viewed%20en%20masse.) page is defined as
    > The spore print is the powdery deposit obtained by allowing spores of a fungal fruit body to fall onto a surface underneath. It is an important diagnostic character in most handbooks for identifying mushrooms. It shows the color of the mushroom spores if viewed en masse.



    <img style="float: left;" src="http://ids-mushroom.appspot.com/images/mushroom%20wireframes_Population.png" width = "600">

    **population** relates to the way the mushroom grows. The image to the left shows examples of three kinds of population. And, finally, **habitat** refers to where the mushroom grows. According to the [Intermountain Herbarium](https://herbarium.usu.edu/fun-with-fungi/collect-and-identify#:~:text=Where%20they%20grow%2C%20such%20as,%2C%20is%20the%20mushrooms'%20substrate.) of the Utah State University:
    > Mushrooms are found almost everywhere, but not all mushrooms are found in all kinds of habitat. Where they grow, such as coniferous forest, oak forest, etc., is the mushrooms' habitat. Some mushrooms develop in only one kind of habitat, such as a bog, a forest, or an open lawn or meadow. What they actually emerge from, such as peat, a log, or soil, is the mushrooms' substrate.

    <span style="color:red">Before going to the data, we want to be very clear: this should not be considered as a guide for mushroom picking. Again, from the [Intermountain Herbarium](https://herbarium.usu.edu/fun-with-fungi/collect-and-identify#:~:text=Where%20they%20grow%2C%20such%20as,%2C%20is%20the%20mushrooms'%20substrate.):
    > People die every year from eating tasty but poisonous mushrooms. There are no so-called tests for telling a poisonous mushroom from a non-poisonous one.
    </span>''', unsafe_allow_html=True
        )

    '## The data frame:'

    st.dataframe(df_mush.head())

    '## Number of unique entries for every column:'
    fig1 = plt.figure(figsize=(12,4), dpi = 400)
    ax = sns.barplot(data=df_mush.describe().transpose().reset_index().sort_values('unique'),x='index',y='unique')
    plt.xticks(rotation=90)
    st.pyplot(fig1)

    '### The main objective is to classify mushrooms as edible or poisonous.'
    '### We can check for class balance in the data set:'
    fig2 = plt.figure(figsize=(8,4), dpi = 400)
    ax = sns.countplot(data = df_mush, y ='class')
    st.pyplot(fig2)

    # Utility function to plot distribution of selected feature
    def get_dists(feature):
        fig, axes = plt.subplots(1, 2, figsize=(21, 6), dpi = 400)
        fig.suptitle('Distribution of '+ feature + ' feature')

        sns.countplot(ax = axes[0], data=df_mush, x=feature, palette="dark",
                      order = df_mush[feature].value_counts().index)
        axes[0].set_title('Distribution')

        sns.countplot(ax = axes[1], data=df_mush, x=feature, hue='class',
                      order = df_mush[feature].value_counts().index)
        axes[1].set_title('Distribution according to class')
        st.pyplot(fig)

    '## Exploring distributions'
    selected_feat = st.selectbox(
        'Select feature to show distribution:',
        df_mush.columns[1:]
    )

    get_dists(selected_feat)

elif selected_view == 'Model away!':
    st.info(
    '''
    ### The modeling will be done using a [decision tree](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html).
    ### Test size and maximum number of leaf nodes can be chosen at the sidebar.
    ''')

    X = df_mush.drop('class', axis = 1)
    y = df_mush['class']
    X = X.drop('veil-type', axis = 1)
    X = pd.get_dummies(X,drop_first=True)

    t_size = st.sidebar.slider(
        'Choose test-size:',
        min_value = 0.1,
        max_value = 0.5,
        value = 0.2,
        step = 0.1
    )

    max_ln = st.sidebar.slider(
        'Choose max_leaf_nodes:',
        min_value = 2,
        max_value = 20,
        value = 5,
        step = 1
    )

    c_v = st.sidebar.slider(
        'Choose number of cross-validation splits:',
        min_value = 2,
        max_value = 20,
        value = 5,
        step = 1
    )

    if st.button('Run model'):

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = t_size, random_state=42)

        model = DecisionTreeClassifier(max_leaf_nodes = max_ln)
        model.fit(X_train, y_train)

        def report_model(model, visual = False):
            model_preds = model.predict(X_test)
            acc = accuracy_score(y_test,model_preds)
            if acc == 1.0:
                st.balloons()
            st.write('Accuracy: ', acc)

            c_report = classification_report(y_test, model_preds)

            '### Confusion matrix:'
            col1, col2 = st.beta_columns(2)
            with col1:
                st.write('')
                st.write('')
                st.write(40*'=')
                st.text('Classification Report:\n ' + c_report)
                st.write(40*'=')
                # plot_confusion_matrix(model, X_test, y_test)
                # st.pyplot()
            with col2:
                cf_matrix = confusion_matrix(y_test, model_preds)
                group_names = ['True Pos','False Neg','False Pos','True Neg']
                categories = ['edible','poisonous']
                group_counts = ["{0:0.0f}".format(value) for value in
                                cf_matrix.flatten()]
                group_percentages = ["{0:.2%}".format(value) for value in
                                     cf_matrix.flatten()/np.sum(cf_matrix)]
                labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
                          zip(group_names,group_counts,group_percentages)]
                labels = np.asarray(labels).reshape(2,2)

                fig_c2 = plt.figure(figsize=(9,9), dpi = 400)
                ax = sns.heatmap(cf_matrix, annot=labels, fmt='',
                                 cmap = 'vlag', xticklabels=categories, yticklabels=categories,
                                 annot_kws={"fontsize":24})
                st.pyplot(fig_c2)

            if visual:
                print('\n')
                '## Visualizing the decision tree:'
                plt.figure(figsize=(12,8),dpi=150)
                plot_tree(model,filled=True,feature_names=X.columns)
                st.pyplot()

        report_model(model, True)

        def get_cv(model, cv):
            st.write('Cross validating the model: ', model)
            st.write(50*'=')
            scores = cross_val_score(model,X_train,y_train, scoring='accuracy',cv=cv)

            st.write('Cross-validated accuracy scores:')
            st.text(scores)
            st.write(50*'=')
            st.write('Mean cross-validated accuracy score:')
            st.write(scores.mean())
            st.write(50*'=' + '\n\n')

        get_cv(model, c_v)
