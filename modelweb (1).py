from fastai import *
import streamlit as st
from fastbook import *
import csv

link_for_train = 'https://www.kaggle.com/code/rayhaank/diabetes-predictor'
#Title
st.title(':violet[ Interactive Report on how different parameters impact diabetes]')
st.subheader("By: Rayhaan Khan")
#good to know
st.subheader("Good to know")
st.markdown("We are using a custom model trained through regression, with help of fast.ai library and publicly availbable NHANES (National Health And Nutrition Examination Survey) data from kaggle. [Click here](%s) for the model training procedure." % link_for_train)

st.markdown("**:blue[First let's identify the high impact factors, i.e. the factors which are the most influential towards the chance of someone (in our case) having diabetes.]**")

#feature importance
st.markdown("**According to research I've done, and running many tests, the most influential ones are:**")
st.markdown("Glucose concentration")
st.markdown("BMI (Body Mass Index)")
st.markdown("Insulin amount")
st.markdown("Blood Pressure (increases insulin resistance)")
st.markdown("Pregnancies (increases insulin resistance)")
st.markdown("Age (one of a major ones)")

gluc = 'https://drive.google.com/file/d/1ur8gpMcK5C09tV6GVFthQXFm8pCiwGs6/view?usp=sharing'
#stats
st.subheader(":orange[Some statistics]")
st.image('srt1.png', caption='Spread over 768 individuals')
st.markdown('[Click here](%s) for more info on glucose test' % gluc)
st.image('srt3.png', caption='Spread over 768 individuals')

st.image('srt2.png', caption='Spread over 768 individuals')

#sliders
st.header(":green[**Lets start changing some parameters**]")
glucose = st.slider("**Glucose amount (mg/dL) [ref. 100]**", 0, 199, 100)
bmi = st.slider("**BMI (Body Mass Index) [ref 30]**", 0, 67, 33)
insulin = st.slider("**Insuliin concentration(2-Hour serum insulin (mu U/ml) [ref. 80-100]**", 0, 846, 122)
bloodp = st.slider("**Blood Pressure (mm Hg)[ref. 80, non preg.]**", 0, 140, 80)
pregnancies = st.slider("**Amount of pregnancies (only for women)**", 0, 14, 2)
SkinThickness = 35
DiabetesPedigreeFunction = 0.627
Age = st.slider("**Age**", 0, 100, 50)

learn = load_learner('model.pkl')

if st.button("press to run", type="primary"):
    with open('diabetes.csv', 'a') as file:
        writer = csv.writer(file)
        writer.writerow([pregnancies, glucose, bloodp, SkinThickness, insulin, bmi, DiabetesPedigreeFunction, Age])
        learn = load_learner('model.pkl')
        test = pd.read_csv('diabetes.csv')
        dl = learn.dls.test_dl(test)
        preds = learn.get_preds(dl=dl)
        preds = [x.item() for x in preds[0]]
        test['Outcome'] = preds
        st.warning('ignore first outcome. Use the second one.')
        st.info(preds)


#st.title("Acknowledgements")
#st.markdown("I would like to say a special thank you to my uncle, Dr. Shadab Khan for providing me with this wonderful opportunity, to learn and expand my knowledge of AI, Deep Learning and ML and always providing valuable feedback for my assignments and projects.")

