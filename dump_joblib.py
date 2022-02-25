import joblib
import sklearn
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# load some categories of newsgroups dataset
categories = [
    "soc.religion.christian",
    "talk.religion.misc",
    "sci.space",
    "comp.sys.mac.hardware",
    'sci.crypt',
    'sci.electronics',
    'sci.med',
    'talk.politics.guns',
    'talk.politics.mideast',
    'comp.graphics',
    
]

newsgropus_training = fetch_20newsgroups(
    subset="train", categories=categories, shuffle=True, random_state=0
)
newsgropus_testing = fetch_20newsgroups(
    subset="test", categories=categories, shuffle=True, random_state=0
)

# make the pipeline

model = make_pipeline(
    TfidfVectorizer(),
    MultinomialNB(),
    )

# train the model
model.fit(newsgropus_training.data, newsgropus_training.target)


# Serielize the model and the target names
model_file = "newsgroups_model.joblib"
model_targets_tuple = (model, newsgropus_training.target_names)
joblib.dump(model_targets_tuple, model_file)