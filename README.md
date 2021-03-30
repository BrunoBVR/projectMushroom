<img src="https://www.thephotoargus.com/wp-content/uploads/2019/01/fungi12.jpg" width = "600">

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
</span>

# The notebook (projectMushroom.ipynb)

Contains basic EDA and modeling with a [decision tree classifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html).

# The app (app/)

Streamlit app to do some data exploration and modeling. The app allows building of a decision tree classifier with controlled number of maximum leaf nodes. The app shows classification report, confusion matrix and a plot of the tree (and some cool balloons if you hit that 100% accuracy!).

Check out the [app](https://share.streamlit.io/brunobvr/projectmushroom/main/my_mushroom_app.py)
