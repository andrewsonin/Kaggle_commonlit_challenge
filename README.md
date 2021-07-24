# Kaggle_commonlit_challenge



**Example of use Aleron's part**


```
nlp = stanza.Pipeline('en', processors="tokenize,mwt,pos,lemma,depparse", model_dir= '~/resourses/')

txt = 'Sey hello to my little NN model, Bro! MIPT is the best place!'

features = text_to_fuatures(txt, model_dir = '../working/stanza-resourses-en/', nlp = nlp)
print(features)
```
